#pragma once

#include <algorithm>

#include "lj.hpp"
#include "system.hpp"

namespace md {

// Adds the force of pair (i, j) to both particles; returns its energy
// (zero beyond the cutoff).
inline double add_pair_force(System& s, std::size_t i, std::size_t j, double rc2) {
    double d[3];
    double r2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        d[k] = min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
        r2 += d[k] * d[k];
    }
    if (r2 >= rc2) return 0.0;
    const double f = lj::pair_force_over_r(r2, rc2);
    s.virial += f * r2;
    for (int k = 0; k < 3; ++k) {
        s.force[3 * i + k] += f * d[k];
        s.force[3 * j + k] -= f * d[k];
    }
    return lj::shifted_pair_energy(r2, rc2);
}

// All-pairs forces with minimum image and cutoff lj::kCutoff.
// Fills s.force and returns the potential energy. Reference implementation.
inline double compute_forces(System& s) {
    const double rc2 = lj::kCutoff * lj::kCutoff;
    std::fill(s.force.begin(), s.force.end(), 0.0);
    s.virial = 0.0;
    double pe = 0.0;
    for (std::size_t i = 0; i < s.n; ++i)
        for (std::size_t j = i + 1; j < s.n; ++j) pe += add_pair_force(s, i, j, rc2);
    return pe;
}

}  // namespace md
