#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace md {

// Flat arrays of length 3*n: x0,y0,z0,x1,y1,z1,...
struct System {
    std::size_t n = 0;
    double box = 0.0;
    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> force;
};

// Wraps a coordinate difference into [-box/2, box/2].
inline double min_image(double d, double box) { return d - box * std::round(d / box); }

inline double kinetic_energy(const System& s) {
    double sum = 0.0;
    for (double v : s.vel) sum += v * v;
    return 0.5 * sum;
}

// Kinetic temperature with 3(N-1) degrees of freedom (momentum is conserved).
inline double temperature(const System& s) {
    return 2.0 * kinetic_energy(s) / (3.0 * static_cast<double>(s.n - 1));
}

}  // namespace md
