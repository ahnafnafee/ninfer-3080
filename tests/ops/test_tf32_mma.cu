#include "ops/common/mma.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace ninfer::test;

// A[16,K] and B[K,8] are ordinary represented FP32 inputs. Both entry points must preserve
// round-to-nearest TF32 accuracy; the bit entry point receives explicitly converted operands.
template <bool BitOperands>
__global__ void product(const float* a, const float* b, float* out, int k) {
    const int lane = threadIdx.x;
    const int row  = lane >> 2;
    const int subk = lane & 3;
    float c[4]     = {};
    for (int base = 0; base < k; base += 8) {
        const float a0 = a[row * k + base + subk];
        const float a1 = a[(row + 8) * k + base + subk];
        const float a2 = a[row * k + base + subk + 4];
        const float a3 = a[(row + 8) * k + base + subk + 4];
        const float b0 = b[(base + subk) * 8 + row];
        const float b1 = b[(base + subk + 4) * 8 + row];
        if constexpr (BitOperands) {
            using ninfer::ops::float_to_tf32_bits;
            ninfer::ops::mma_tf32_bits(c[0], c[1], c[2], c[3], float_to_tf32_bits(a0),
                                       float_to_tf32_bits(a1), float_to_tf32_bits(a2),
                                       float_to_tf32_bits(a3), float_to_tf32_bits(b0),
                                       float_to_tf32_bits(b1));
        } else {
            ninfer::ops::mma_tf32(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
        }
    }
    out[row * 8 + subk * 2]           = c[0];
    out[row * 8 + subk * 2 + 1]       = c[1];
    out[(row + 8) * 8 + subk * 2]     = c[2];
    out[(row + 8) * 8 + subk * 2 + 1] = c[3];
}

int check_product(int k, int fixture, bool bit_operands) {
    std::vector<float> a(16 * k), b(k * 8), got(128);
    std::mt19937 generator(317 + k + fixture);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    std::uniform_int_distribution<int> exponent(-20, 12);
    const bool exact_a = fixture == 0 || fixture == 1;
    const bool exact_b = fixture == 0 || fixture == 2;
    const auto fill    = [&](std::vector<float>& input, bool exact) {
        for (float& value : input) {
            // These positive values expose truncation bias: 3/4 of a TF32 step above one.
            value = fixture < 4 ? 1.000732421875f : values(generator);
            if (fixture == 5) { value = std::ldexp(value, exponent(generator)); }
        }
        if (exact) { round_to_bf16(input); }
    };
    fill(a, exact_a);
    fill(b, exact_b);
    GuardedDeviceBuffer da(a.size() * sizeof(float)), db(b.size() * sizeof(float));
    GuardedDeviceBuffer dout(got.size() * sizeof(float));
    da.copy_from_host(a.data(), a.size() * sizeof(float));
    db.copy_from_host(b.data(), b.size() * sizeof(float));
    if (bit_operands) {
        product<true><<<1, 32>>>(static_cast<const float*>(da.data()),
                                 static_cast<const float*>(db.data()),
                                 static_cast<float*>(dout.data()), k);
    } else {
        product<false><<<1, 32>>>(static_cast<const float*>(da.data()),
                                  static_cast<const float*>(db.data()),
                                  static_cast<float*>(dout.data()), k);
    }
    cuda_check_last_launch("TF32 product");
    dout.copy_to_host(got.data(), got.size() * sizeof(float));

    // Independent FP64 oracle over the actual public operands, with an error bound derived
    // from nearest TF32 operand rounding and FP32 accumulation. BF16 inputs are TF32-exact.
    const double ua           = exact_a ? 0.0 : 0x1p-11;
    const double ub           = exact_b ? 0.0 : 0x1p-11;
    const double accumulation = 4.0 * k * std::numeric_limits<float>::epsilon();
    int failures              = 0;
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 8; ++col) {
            double expected     = 0.0;
            double absolute_sum = 0.0;
            for (int index = 0; index < k; ++index) {
                const double term = static_cast<double>(a[row * k + index]) * b[index * 8 + col];
                expected += term;
                absolute_sum += std::abs(term);
            }
            const double bound    = (ua + ub + ua * ub + accumulation) * absolute_sum;
            const double observed = got[row * 8 + col];
            if (!std::isfinite(observed) || std::abs(observed - expected) > bound) {
                if (++failures == 1) {
                    std::cerr << "TF32 precision: K=" << k << " fixture=" << fixture
                              << " bit_operands=" << bit_operands
                              << " error=" << observed - expected << " bound=" << bound << '\n';
                }
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        int failures = 0;
        for (int k : {8, 64, 128}) {
            for (int fixture = 0; fixture < 6; ++fixture) {
                failures += check_product(k, fixture, false);
                failures += check_product(k, fixture, true);
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TF32 MMA precision\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TF32 MMA test: " << error.what() << '\n';
        return 1;
    }
}
