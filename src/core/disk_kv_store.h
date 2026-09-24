#pragma once

// A file-backed, self-describing, LRU page store: the disk (L3) tier under the Host KV arena.
//
// The data file is an array of 4 KiB-aligned slots. Each slot starts with a 48-byte header (magic,
// content identity, CRC-32 of the payload, LRU stamp), so the identity-to-slot index can always be
// rebuilt by scanning the slots. The index file (<path>.idx) is replaced atomically (temporary
// file and rename) whenever the live set changes; there is no fsync, so a crash can lose the last
// unpublished pages but never makes a stale row trusted: every index row is checked against its
// slot header on open, and every read verifies the payload CRC.
//
// Self-contained: no CUDA and no engine types.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer {

// Content identity of one stored page: the engine's rolling prefix digest at the page's last
// token, the engine configuration tag, and that frontier.
struct DiskKVIdentity {
    std::uint64_t lo       = 0;
    std::uint64_t hi       = 0;
    std::uint32_t tag      = 0;
    std::uint32_t frontier = 0;

    [[nodiscard]] friend constexpr bool operator==(const DiskKVIdentity&,
                                                   const DiskKVIdentity&) noexcept = default;
};

struct DiskKVIdentityHash {
    std::size_t operator()(const DiskKVIdentity& id) const noexcept {
        std::uint64_t h = id.lo;
        h ^= id.hi + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        h ^= std::uint64_t{id.tag} * 0xff51afd7ed558ccdULL;
        h ^= std::uint64_t{id.frontier} * 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 31U;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33U;
        return static_cast<std::size_t>(h);
    }
};

class DiskKVStore {
public:
    struct Options {
        std::string path; // data file; created if absent, never truncated
        std::size_t slot_size      = 0;
        std::size_t capacity_bytes = 0;
        std::uint32_t max_slots    = 0; // derived from capacity_bytes when zero
        bool verify_crc            = true;
        // Defer publishing new pages to the index until flush_index(). Evictions stay eager. The
        // owner must flush before shutdown; the destructor does not.
        bool defer_index_updates = false;
    };

    // Opens or creates the data file and loads (or rebuilds) the index. Throws on I/O errors or a
    // data file laid out for a different geometry.
    explicit DiskKVStore(Options options);
    ~DiskKVStore();

    DiskKVStore(const DiskKVStore&)            = delete;
    DiskKVStore& operator=(const DiskKVStore&) = delete;

    [[nodiscard]] std::size_t slot_size() const noexcept { return options_.slot_size; }

    [[nodiscard]] std::uint32_t slot_count() const noexcept { return max_slots_; }

    [[nodiscard]] std::uint32_t live_slots() const;
    [[nodiscard]] std::size_t used_bytes() const;

    [[nodiscard]] const std::string& path() const noexcept { return options_.path; }

    [[nodiscard]] bool index_rebuilt_from_scan() const noexcept { return rebuilt_from_scan_; }

    // Inserts the page, or refreshes its LRU stamp when already present. A full store evicts its
    // least recently used page first (reported through `evicted`). False when the store is full
    // and every live slot is being read.
    bool upsert_page(const DiskKVIdentity& id, std::span<const std::byte> bytes,
                     std::vector<DiskKVIdentity>* evicted = nullptr);

    // Copies the page into `destination` (slot_size bytes) after checking identity and CRC, and
    // refreshes its LRU stamp. The copy and CRC run outside the store lock; eviction skips a slot
    // while it is being read.
    bool read_page(const DiskKVIdentity& id, std::span<std::byte> destination);

    [[nodiscard]] bool contains(const DiskKVIdentity& id) const;
    bool touch(const DiskKVIdentity& id);
    // Drops the page if it is live and not being read.
    bool evict(const DiskKVIdentity& id);
    [[nodiscard]] std::vector<DiskKVIdentity> live_identities() const;

    // Publishes pending index changes (atomic replacement, not a durability barrier).
    void flush_index();

    static constexpr std::size_t kSlotHeaderSize = 48;
    // CRC-32 with the reflected 0xEDB88320 polynomial (zlib's), computed slicing-by-8.
    [[nodiscard]] static std::uint32_t crc32(std::span<const std::byte> data) noexcept;

private:
    struct Header {
        std::uint32_t magic                   = 0;
        std::uint32_t reserved                = 0;
        std::uint64_t last_used               = 0;
        std::uint64_t lo                      = 0;
        std::uint64_t hi                      = 0;
        std::uint32_t tag                     = 0;
        std::uint32_t frontier                = 0;
        std::uint32_t crc                     = 0;
        std::uint32_t pad                     = 0;
        static constexpr std::uint32_t kMagic = 0x4E44564B;
    };

    static_assert(sizeof(Header) == kSlotHeaderSize);

    struct IndexRow {
        std::uint64_t lo       = 0;
        std::uint64_t hi       = 0;
        std::uint32_t tag      = 0;
        std::uint32_t frontier = 0;
        std::uint32_t slot     = 0;
        std::uint32_t pad      = 0;
    };

    struct IndexHeader {
        std::uint32_t magic                   = 0;
        std::uint32_t version                 = 0;
        std::uint32_t count                   = 0;
        std::uint32_t slot_size               = 0;
        std::uint32_t max_slots               = 0;
        std::uint32_t pad                     = 0;
        std::uint64_t clock                   = 0;
        static constexpr std::uint32_t kMagic = 0x4944564B;
    };

    [[nodiscard]] std::size_t slot_pitch() const noexcept;
    [[nodiscard]] std::size_t payload_offset(std::uint32_t slot) const noexcept;
    [[nodiscard]] Header& header(std::uint32_t slot) const noexcept;
    [[nodiscard]] static bool matches(const Header& header, const DiskKVIdentity& id) noexcept;

    void map_file(bool create);
    void unmap_file() noexcept;
    bool load_index();
    void rebuild_from_scan();
    void persist_index_locked();
    void clear_slot(std::uint32_t slot) noexcept;
    // Evicts the least recently used live slot nobody is reading. Caller holds mutex_.
    std::optional<DiskKVIdentity> evict_one_lru_locked();

    mutable std::mutex mutex_;
    Options options_;
    std::byte* base_           = nullptr;
    std::size_t file_bytes_    = 0;
    std::uint32_t max_slots_   = 0;
    std::uint64_t clock_       = 0;
    bool rebuilt_from_scan_    = false;
    bool index_dirty_          = false;
    std::intptr_t file_handle_ = -1;
#ifdef _WIN32
    std::intptr_t mapping_handle_ = -1;
#endif
    std::unordered_map<DiskKVIdentity, std::uint32_t, DiskKVIdentityHash> index_;
    std::vector<std::uint32_t> free_slots_;
    std::vector<std::uint32_t> readers_;
};

} // namespace ninfer
