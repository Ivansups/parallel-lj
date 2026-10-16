#pragma once

namespace lj {

inline constexpr double kCutoff = 2.5;

// Pair energy U(r) = 4(r^-12 - r^-6) in reduced units; r2 is the squared distance.
inline double pair_energy(double r2, double rc2) {
    if (r2 >= rc2) return 0.0;
    const double inv6 = 1.0 / (r2 * r2 * r2);
    return 4.0 * inv6 * (inv6 - 1.0);
}

// Energy shifted by U(rc) so it is continuous at the cutoff (needed for NVE conservation).
inline double shifted_pair_energy(double r2, double rc2) {
    if (r2 >= rc2) return 0.0;
    const double inv6c = 1.0 / (rc2 * rc2 * rc2);
    return pair_energy(r2, rc2) - 4.0 * inv6c * (inv6c - 1.0);
}

// F(r)/r, so the force vector is this value times the displacement.
inline double pair_force_over_r(double r2, double rc2) {
    if (r2 >= rc2) return 0.0;
    const double inv2 = 1.0 / r2;
    const double inv6 = inv2 * inv2 * inv2;
    return 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0);
}

}  // namespace lj
