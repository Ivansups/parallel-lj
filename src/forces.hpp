#pragma once

#include <algorithm>

#include "lj.hpp"
#include "system.hpp"

namespace md {

// All-pairs forces with minimum image and cutoff lj::kCutoff.
// Fills s.force and returns the potential energy.
inline double compute_forces(System& s) {
    const double rc2 = lj::kCutoff * lj::kCutoff;
    std::fill(s.force.begin(), s.force.end(), 0.0);
    double pe = 0.0;
    for (std::size_t i = 0; i < s.n; ++i)
        for (std::size_t j = i + 1; j < s.n; ++j) {
            double d[3];
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                d[k] = min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
                r2 += d[k] * d[k];
            }
            if (r2 >= rc2) continue;
            pe += lj::shifted_pair_energy(r2, rc2);
            const double f = lj::pair_force_over_r(r2, rc2);
            for (int k = 0; k < 3; ++k) {
                s.force[3 * i + k] += f * d[k];
                s.force[3 * j + k] -= f * d[k];
            }
        }
    return pe;
}

}  // namespace md
