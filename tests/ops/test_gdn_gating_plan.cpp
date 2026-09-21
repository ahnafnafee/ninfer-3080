// Which schedule the BF16 GDN gating projection gets for a width, on devices with different numbers
// of multiprocessors.
//
// The route table was tuned on an 82-SM RTX 3090. Its cooperative schedules need their whole grid
// resident, and a grid that exceeds what a device holds is rejected by the driver rather than merely
// slow. A 48-SM RTX A4000 therefore cannot take the tuned route for the widths near the top of each
// cooperative range; it must fall through to a less-split schedule. This checks both halves: the
// tuned device keeps exactly its tuned routes, and a smaller device never gets a grid it cannot hold.

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::Bf16GdnGatingPlan;
using ninfer::ops::detail::Bf16GdnGatingProblem;
using ninfer::ops::detail::Bf16GdnGatingScheduleId;
using Id = Bf16GdnGatingScheduleId;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

struct Geometry {
    const char* label;
    std::int32_t heads;
    std::int32_t rows;
    std::int32_t tile_cols;
    std::int32_t row_tiles;
};

constexpr Geometry k27{"27B", 48, 5120, 128, 3};
constexpr Geometry k35{"35B-A3B", 32, 2048, 64, 2};

// The tuned table on 82 SMs, restated from the tuning notes rather than read back from the code.
Id tuned(const Geometry& g, std::int32_t cols) {
    if (g.heads == 48) {
        if (cols == 1) { return Id::GemvPairedRows; }
        if (cols <= 8) { return Id::SmallTSplit10; }
        if (cols <= 768) { return Id::MmaCooperativeSplit8; }
        if (cols <= 1664) { return Id::MmaCooperativeSplit4; }
        if (cols <= 3456) { return Id::MmaCooperativeSplit2; }
        return Id::MmaUnsplit;
    }
    if (cols <= 127) { return Id::MmaCooperativeSplit16; }
    if (cols <= 960) { return Id::MmaCooperativeSplit8; }
    if (cols <= 1920) { return Id::MmaCooperativeSplit4; }
    if (cols <= 3904) { return Id::MmaCooperativeSplit2; }
    return Id::MmaUnsplit;
}

std::int32_t split_of(Id schedule) {
    switch (schedule) {
    case Id::MmaCooperativeSplit32: return 32;
    case Id::MmaCooperativeSplit16: return 16;
    case Id::MmaCooperativeSplit8: return 8;
    case Id::MmaCooperativeSplit4: return 4;
    case Id::MmaCooperativeSplit2: return 2;
    default: return 0; // not cooperative
    }
}

// Resident CTAs per SM of each cooperative schedule, as measured on the sm_86 build.
std::int32_t ctas_per_sm(const Geometry& g, Id schedule) {
    if (g.heads == 48) { return 2; }
    if (schedule == Id::MmaCooperativeSplit32) { return 2; }
    if (schedule == Id::MmaCooperativeSplit16) { return 4; }
    return 3;
}

bool grid_fits(const Geometry& g, Id schedule, std::int32_t cols, std::int32_t sms) {
    const std::int32_t split = split_of(schedule);
    if (split == 0) { return true; }
    const std::int64_t grid =
        static_cast<std::int64_t>((cols + g.tile_cols - 1) / g.tile_cols) * g.row_tiles * split;
    return grid <= static_cast<std::int64_t>(ctas_per_sm(g, schedule)) * sms;
}

} // namespace

int main() {
    constexpr std::int32_t kWidths = 6000;
    for (const Geometry& g : {k27, k35}) {
        // The tuned device keeps its tuned routes at every width.
        for (std::int32_t cols = 1; cols <= kWidths; ++cols) {
            const Bf16GdnGatingPlan plan =
                ninfer::ops::detail::bf16_gdn_gating_resolve_plan({g.heads, g.rows, cols}, 82);
            check(plan.schedule == tuned(g, cols),
                  std::string(g.label) + ": 82 SMs changed the tuned route at " +
                      std::to_string(cols) + " columns");
        }
        // A device with fewer SMs never gets a grid it cannot hold, and only leaves the tuned route
        // when that route does not fit, moving to a schedule that splits K no more.
        for (const std::int32_t sms : {84, 68, 48, 40, 30}) {
            for (std::int32_t cols = 1; cols <= kWidths; ++cols) {
                const Bf16GdnGatingPlan plan = ninfer::ops::detail::bf16_gdn_gating_resolve_plan(
                    {g.heads, g.rows, cols}, sms);
                const std::string where = std::string(g.label) + ": " + std::to_string(sms) +
                                          " SMs, " + std::to_string(cols) + " columns";
                check(grid_fits(g, plan.schedule, cols, sms), where + " got a grid that is not resident");
                const Id preferred = tuned(g, cols);
                if (grid_fits(g, preferred, cols, sms)) {
                    check(plan.schedule == preferred, where + " left a tuned route that fits");
                } else {
                    check(split_of(plan.schedule) < split_of(preferred),
                          where + " did not move to a less-split schedule");
                }
            }
        }
        // Naming a schedule that does not fit is an error, not a silent substitution.
        bool threw = false;
        try {
            (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
                Id::MmaCooperativeSplit8, {g.heads, g.rows, 1024}, 30);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, std::string(g.label) + ": an over-large candidate was accepted");
    }
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "gdn gating plan holds across SM counts\n";
    return 0;
}
