#pragma once

#include <cmath>
#include <random>

#include "system.hpp"

namespace md {

// FCC lattice of 4*cells^3 particles at density rho; Maxwell velocities with
// zero total momentum, rescaled to exactly the requested temperature.
inline System make_fcc(int cells, double rho, double temp, unsigned seed) {
    System s;
    s.n = 4u * static_cast<std::size_t>(cells) * cells * cells;
    s.box = std::cbrt(static_cast<double>(s.n) / rho);
    const double a = s.box / cells;
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};

    s.pos.reserve(3 * s.n);
    for (int i = 0; i < cells; ++i)
        for (int j = 0; j < cells; ++j)
            for (int k = 0; k < cells; ++k)
                for (const auto& b : basis) {
                    s.pos.push_back((i + b[0]) * a);
                    s.pos.push_back((j + b[1]) * a);
                    s.pos.push_back((k + b[2]) * a);
                }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    s.vel.resize(3 * s.n);
    for (double& v : s.vel) v = gauss(rng);

    double mean[3] = {0, 0, 0};
    for (std::size_t i = 0; i < s.n; ++i)
        for (int d = 0; d < 3; ++d) mean[d] += s.vel[3 * i + d];
    for (std::size_t i = 0; i < s.n; ++i)
        for (int d = 0; d < 3; ++d) s.vel[3 * i + d] -= mean[d] / static_cast<double>(s.n);

    const double scale = std::sqrt(temp / temperature(s));
    for (double& v : s.vel) v *= scale;

    s.force.assign(3 * s.n, 0.0);
    return s;
}

}  // namespace md
