#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// CUDA function attributes are scoped to a device context. A process-wide `static` result
// therefore leaves the same kernel unconfigured the first time it launches on a second GPU, which
// shows up as a launch failure or silently wrong output rather than as a clear error. Give each
// launcher specialization a cheap, device-keyed cache instead.
//
// Single-GPU behaviour is unchanged: the map holds exactly one entry.
template <typename Configure>
void configure_cuda_device_once(Configure&& configure) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaError_t> results;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaError_t result = cudaSuccess;
    {
        const std::scoped_lock lock(mutex);
        const auto existing = results.find(device);
        if (existing != results.end()) {
            result = existing->second;
        } else {
            result = std::forward<Configure>(configure)();
            results.emplace(device, result);
        }
    }
    CUDA_CHECK(result);
}

// How many pieces one cross-rank transfer is split into.
//
// A crossing is a serial D2H then H2D, so the two halves never overlap: a 4 MB residual stream
// costs ~0.765 ms measured, against the ~0.33 ms its bandwidth alone implies. Splitting the byte
// range lets piece i+1 stream out of the source while piece i streams into the destination. It is
// a pure byte-level pipeline -- the same bytes in the same order -- so it cannot affect which
// kernels run or what they compute.
inline constexpr std::size_t kCrossingPipelineDepth = 4;

// Default pinned staging capacity: covers a 1024-token residual crossing (hidden 5120, BF16)
// with generous headroom. A caller that configures a larger prefill chunk than this covers
// should size the crossing staging buffer explicitly at construction instead of relying on this.
inline constexpr std::size_t kDefaultCrossingStagingBytes = 64ULL << 20;

// The most ranks one context holds. Layer pipelines and tensor groups are both bounded well below
// this; the limit exists so a mistyped device list fails at construction.
inline constexpr std::size_t kMaxRanks = 8;

// Makes a CUDA device the calling thread's current device for a scope, and restores whatever was
// current before. It carries no state of its own, so nothing else can be left pointing at the wrong
// device by an exception or an early return.
class DeviceBinding {
public:
    explicit DeviceBinding(int device);
    ~DeviceBinding() noexcept;

    DeviceBinding(const DeviceBinding&)            = delete;
    DeviceBinding& operator=(const DeviceBinding&) = delete;

private:
    int previous_ = 0;
    bool changed_ = false;
};

// One device's execution resources inside a DeviceContext: its streams, fence and properties.
// Everything a rank owns is fixed at construction. Ranks may name the same physical device (a test
// mode that exercises the multi-rank paths on one card), in which case they still own distinct
// streams.
struct RankContext {
    std::size_t index = 0;
    int device        = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaEvent_t fence          = nullptr;
    cudaDeviceProp props{};

    [[nodiscard]] int compute_capability() const noexcept {
        return props.major * 10 + props.minor;
    }
    // Streaming-multiprocessor count. Distinct from compute_capability(): every sm_86 part shares
    // capability 86 but not this count (RTX 3090 has 82, A4000 has 48), so any device-wide
    // residency or launch budget must read this.
    [[nodiscard]] int multiprocessor_count() const noexcept { return props.multiProcessorCount; }
    [[nodiscard]] std::size_t total_vram() const noexcept { return props.totalGlobalMem; }
    [[nodiscard]] DeviceExecutionView execution_view() const noexcept {
        return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
    }

    // State of the previous crossing protocol (`stage_cross_rank_copy`). It goes with the expert
    // offload path; `StageLink` keeps its own fences.
    std::array<cudaEvent_t, kCrossingPipelineDepth> piece_fences{};
    std::array<cudaEvent_t, kCrossingPipelineDepth> piece_consumed{};
    // Where each consumed fence was last recorded: outside any capture, and the capture it
    // belonged to. Both are kept because a graph's event-record node only takes effect when
    // the graph is launched, so an earlier eager record remains the event's real state.
    std::array<bool, kCrossingPipelineDepth> piece_consumed_eager{};
    std::array<unsigned long long, kCrossingPipelineDepth> piece_consumed_capture{};
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    // One entry keeps the single-device route. Several entries open one rank per entry for
    // model-parallel execution, up to kMaxRanks: matching compute capability is required across
    // all of them, before any weight is uploaded. Peer access is only probed and recorded as a
    // capability -- it is not required, since transfers stage through pinned host memory when it is
    // unavailable.
    //
    // `min_crossing_staging_bytes` sizes the pinned cross-rank staging buffer (see
    // `crossing_staging()`); the default covers only a modest residual crossing. A caller that
    // configures a larger prefill chunk must size this explicitly, since one crossing has to fit
    // in a single staged transfer.
    explicit DeviceContext(std::span<const int> device_ids,
                           std::size_t min_crossing_staging_bytes = kDefaultCrossingStagingBytes);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    // Streaming-multiprocessor count of the attached device. Distinct from compute_capability():
    // every sm_86 part shares capability 86 but not this count (RTX 3090 has 82, RTX 3090 Ti has
    // 84), so any device-wide residency budget must read this, not compute_capability().
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool model_parallel() const noexcept;
    // Rank `index`'s resources. Throws std::out_of_range past the end.
    [[nodiscard]] const RankContext& rank(std::size_t index) const;
    // The rank that takes input (embedding, ingress) and the rank that produces output (head,
    // sampling, egress). Both are rank 0 on one device.
    [[nodiscard]] const RankContext& entry() const { return rank(0); }
    [[nodiscard]] const RankContext& head() const { return rank(size() - 1); }
    // Whether two ranks are the same physical device.
    [[nodiscard]] bool same_physical_device(std::size_t a, std::size_t b) const;
    // True when the devices can DMA directly to each other. False is not an error: cudaMemcpyPeer
    // still works, staging through host memory at roughly 13us per hop instead of a couple. Only
    // consult this to pick between schedules -- a design crossing once per token does not care,
    // one crossing twice per layer does. The no-argument form is true only when every pair of
    // ranks can reach each other.
    [[nodiscard]] bool peer_access() const noexcept;
    [[nodiscard]] bool peer_access(std::size_t from, std::size_t to) const;
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept;
    [[nodiscard]] cudaStream_t stream_for_rank(std::size_t rank) const;
    // Weight upload goes to the arena of the rank that owns the object, and a host-to-device copy
    // has to be issued on a stream belonging to the destination device.
    [[nodiscard]] cudaStream_t transfer_stream_for_rank(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t fence_for_rank(std::size_t rank) const;
    // Pinned host staging for cross-rank copies. Allocated only for a model-parallel context.
    //
    // cudaMemcpyPeerAsync is the obvious way to move a tensor between ranks and is the wrong one
    // here on two counts, both measured on a bridgeless 2x 3090: it runs at roughly half the rate
    // of an explicit D2H/H2D pair through pinned host (0.69 ms against 0.33 ms for 4 MB), and it
    // cannot be captured into a CUDA graph at all -- capture fails with
    // cudaErrorStreamCaptureUnsupported, whereas a memcpy to or from pinned host is an ordinary
    // graph node. Losing capture costs prefill a factor of 3.4, which dwarfs the transfer itself,
    // so being capturable matters far more than the copy rate.
    [[nodiscard]] void* crossing_staging() const noexcept;
    // Fence for one piece of a pipelined cross-rank transfer.
    [[nodiscard]] cudaEvent_t piece_fence(std::size_t rank, std::size_t piece) const;
    // Recorded on the destination stream once a staged piece has been read out of the shared
    // crossing buffer. The next crossing waits on it before overwriting that piece, which is what
    // keeps one pinned buffer safe across back-to-back crossings.
    [[nodiscard]] cudaEvent_t piece_consumed_fence(std::size_t rank, std::size_t piece) const;
    // Whether that fence may be waited on from work whose capture id is `capture_id` (0 for work
    // outside any capture). During capture, a wait on an event whose last record was not part of
    // the same capture fails with cudaErrorStreamCaptureIsolation -- and the crossing path exists
    // to be captured -- so the first crossing inside a graph has nothing to wait for and says so
    // here. Its safety comes from the graph instead: the captured crossings join back into the
    // origin stream, so one launch's H2Ds all complete before the next launch's D2Hs begin.
    //
    // The one thing this cannot express is an eager crossing still in flight when a graph holding
    // crossings is launched, since a launch cannot wait on an eager fence without breaking the
    // capture. Callers capture while the decoder is idle and launch afterwards, which is the only
    // order this path is used in.
    [[nodiscard]] bool piece_consumed_visible(std::size_t rank, std::size_t piece,
                                              unsigned long long capture_id) const;
    // Records that the fence for this piece has just been recorded by work with that capture id.
    void note_piece_consumed(std::size_t rank, std::size_t piece, unsigned long long capture_id);
    [[nodiscard]] std::size_t crossing_staging_bytes() const noexcept;
    void activate_rank(std::size_t rank);
    void synchronize_rank(std::size_t rank) const;
    void synchronize() const;
    int sm() const noexcept;

private:
    void refresh_active_aliases() noexcept;
    void release() noexcept;

    void* crossing_staging_             = nullptr;
    std::size_t crossing_staging_bytes_ = 0;
    std::vector<RankContext> endpoints_;
    std::vector<int> device_ids_;
    // peer_matrix_[from * size + to]: whether `from` can DMA into `to`. A rank always reaches a
    // rank on its own physical device.
    std::vector<char> peer_matrix_;
    std::size_t active_rank_ = 0;
    bool peer_access_        = false;
};

// Makes rank `rank`'s device current for a scope and restores the previous current device.
// Unlike ScopedDeviceRank it does not touch the context's active-rank aliases, so it is safe
// anywhere a context is only read.
class RankBinding {
public:
    RankBinding(const DeviceContext& context, std::size_t rank)
        : binding_(context.rank(rank).device) {}

private:
    DeviceBinding binding_;
};

// One CUDA stream per rank, indexed by rank. An object whose memory lives on several ranks takes one
// of these, so each piece of work goes on the stream of the device that owns the memory it touches.
//
// A single stream converts implicitly, which is what a single-rank object wants. Handing one to an
// object that spans several ranks fails on the first rank past 0 rather than silently issuing that
// rank's copies on the wrong device's stream.
class RankStreams {
public:
    RankStreams(cudaStream_t single = nullptr) noexcept : count_(1) { streams_[0] = single; }
    explicit RankStreams(std::span<const cudaStream_t> streams) : count_(streams.size()) {
        if (streams.empty() || streams.size() > kMaxRanks) {
            throw std::invalid_argument("RankStreams needs between one and kMaxRanks streams");
        }
        std::copy(streams.begin(), streams.end(), streams_.begin());
    }
    // Every rank's compute stream, or every rank's transfer stream.
    [[nodiscard]] static RankStreams compute(const DeviceContext& context);
    [[nodiscard]] static RankStreams transfer(const DeviceContext& context);

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] cudaStream_t operator[](std::size_t rank) const {
        if (rank >= count_) {
            throw std::out_of_range("no stream was given for rank " + std::to_string(rank) +
                                    " (" + std::to_string(count_) + " given)");
        }
        return streams_[rank];
    }

private:
    std::array<cudaStream_t, kMaxRanks> streams_{};
    std::size_t count_ = 0;
};

inline RankStreams RankStreams::compute(const DeviceContext& context) {
    std::array<cudaStream_t, kMaxRanks> streams{};
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        streams[rank] = context.rank(rank).stream;
    }
    return RankStreams(std::span<const cudaStream_t>(streams.data(), context.size()));
}

inline RankStreams RankStreams::transfer(const DeviceContext& context) {
    std::array<cudaStream_t, kMaxRanks> streams{};
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        streams[rank] = context.rank(rank).transfer_stream;
    }
    return RankStreams(std::span<const cudaStream_t>(streams.data(), context.size()));
}

// Binds a rank for the duration of a scope and restores the previous one, so a caller that has to
// touch the secondary device cannot leave the thread bound to it.
class ScopedDeviceRank {
public:
    ScopedDeviceRank(DeviceContext& context, std::size_t rank);
    ~ScopedDeviceRank() noexcept;

    ScopedDeviceRank(const ScopedDeviceRank&)            = delete;
    ScopedDeviceRank& operator=(const ScopedDeviceRank&) = delete;

private:
    DeviceContext& context_;
    std::size_t previous_rank_ = 0;
};

// Moves `bytes` from `source` on `from_rank` to `destination` on `to_rank` through the shared
// pinned crossing buffer, pipelined over `kCrossingPipelineDepth` pieces and ordered entirely by
// events, so the whole transfer stays capturable into a CUDA graph.
//
// Callers with both ranks on one physical device should copy device-to-device instead; this path
// exists for a genuine two-card crossing. It lives here rather than in the decoder so a test can
// drive it with two ranks pinned to one device and check the fence protocol without a second card.
void stage_cross_rank_copy(DeviceContext& context, const void* source, std::size_t from_rank,
                           void* destination, std::size_t to_rank, std::size_t bytes);

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
