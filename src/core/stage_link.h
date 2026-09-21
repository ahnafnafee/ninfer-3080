#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <vector>

namespace ninfer {

// One-way, capture-safe transfer of a payload from a rank to another rank through a small ring of
// slots. It is the boundary primitive of a layer pipeline and the substrate of a tensor-parallel
// exchange.
//
// A transfer has four independent halves, and keeping them apart is the point of the class:
//
//   send_payload / recv_payload   plain memcpy nodes. Legal inside stream capture, so each stage's
//                                 graph carries its own half and no event ever crosses a graph.
//   mark_* / wait_*               event records and waits. Eager only. A wait on an event whose last
//                                 record was outside the capture fails with
//                                 cudaErrorStreamCaptureIsolation, which is the trap the previous
//                                 crossing protocol had to track capture ids to avoid. Here the
//                                 fences are simply issued between graph launches.
//
// Ordering for one slot is: sender `wait_drained`, `send_payload`, `mark_filled`; receiver
// `wait_filled`, `recv_payload`, `mark_drained`. The eager `send` and `recv` do exactly that (and
// pipeline large payloads in pieces); a graph-based caller issues the halves itself around its graph
// launches. A slot must not be reused until the receiver's `mark_drained` for the previous use has
// been enqueued, which is what `wait_drained` enforces; a ring of two or more slots lets one
// transfer fill while the previous one drains.
//
// Two transports, chosen once from the topology:
//   StagedHost  D2H into a pinned host slot, then H2D on the receiver. Works between any two cards,
//               peer access or not, and is an ordinary graph node on both sides.
//   Local       both ranks are the same physical device: a device-to-device copy through a device
//               slot. Tests set `force_staged` to run the real protocol on one card.
//
// No direct peer transport yet: cudaMemcpyPeerAsync cannot be captured, and no card measured so far
// (2x 3090 and 2x A4000, PCIe) exposes peer access to justify writing one.
enum class LinkTransport { StagedHost, Local };

struct StageLinkOptions {
    // Largest payload one slot carries.
    std::size_t slot_bytes = 0;
    std::size_t slots      = 2;
    // Use the staged protocol even when both ranks share a physical device.
    bool force_staged = false;
};

// How many pieces an eager transfer is split into so the D2H of one piece overlaps the H2D of the
// previous one. A pure byte-level pipeline: same bytes, same order.
inline constexpr std::size_t kLinkPipelineDepth = 4;
// Below this the per-piece fence overhead costs more than the overlap saves.
inline constexpr std::size_t kLinkMinimumPipelinedBytes = 256U << 10;

class StageLink {
public:
    // `from_rank` and `to_rank` index into `context`; they may name the same physical device.
    StageLink(const DeviceContext& context, std::size_t from_rank, std::size_t to_rank,
              StageLinkOptions options);
    ~StageLink();

    StageLink(const StageLink&)            = delete;
    StageLink& operator=(const StageLink&) = delete;
    StageLink(StageLink&& other) noexcept;
    StageLink& operator=(StageLink&& other) noexcept;

    [[nodiscard]] LinkTransport transport() const noexcept { return transport_; }
    [[nodiscard]] std::size_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::size_t from_rank() const noexcept { return from_rank_; }
    [[nodiscard]] std::size_t to_rank() const noexcept { return to_rank_; }

    // Payload halves. Both are single copies and safe to capture.
    void send_payload(const void* source, std::size_t bytes, std::size_t slot,
                      cudaStream_t from_stream);
    void recv_payload(void* destination, std::size_t bytes, std::size_t slot,
                      cudaStream_t to_stream);

    // Fence halves. Eager only: never call these on a stream that is capturing.
    void wait_drained(std::size_t slot, cudaStream_t from_stream);
    void mark_filled(std::size_t slot, cudaStream_t from_stream);
    void wait_filled(std::size_t slot, cudaStream_t to_stream);
    void mark_drained(std::size_t slot, cudaStream_t to_stream);

    // Eager conveniences composing the halves. A payload of kLinkMinimumPipelinedBytes or more is
    // split into kLinkPipelineDepth pieces with a fence each, so the two copies overlap.
    void send(const void* source, std::size_t bytes, std::size_t slot, cudaStream_t from_stream);
    void recv(void* destination, std::size_t bytes, std::size_t slot, cudaStream_t to_stream);

private:
    struct Slot {
        void* data = nullptr;
        std::array<cudaEvent_t, kLinkPipelineDepth> filled{};
        std::array<cudaEvent_t, kLinkPipelineDepth> drained{};
        // Pieces the most recent send used. The receiver's `mark_drained` records the same count.
        std::size_t pieces = 1;
    };

    void check_slot(std::size_t slot, std::size_t bytes) const;
    void release() noexcept;

    std::size_t from_rank_       = 0;
    std::size_t to_rank_         = 0;
    int from_device_             = 0;
    int to_device_               = 0;
    LinkTransport transport_     = LinkTransport::StagedHost;
    std::size_t slots_           = 0;
    std::size_t slot_bytes_      = 0;
    std::vector<Slot> ring_;
};

} // namespace ninfer
