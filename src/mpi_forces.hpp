#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "lj.hpp"
#include "mpi_exchange.hpp"

namespace md::mpi {

// Per-rank contributions; sum them over ranks for the global values.
struct LocalSums {
    double potential = 0.0;  // shifted potential energy
    double virial = 0.0;
    double pairs = 0.0;      // pairs inside the cutoff (owned-ghost pairs count 1/2)
    double kinetic = 0.0;
};

namespace detail {

inline double local_kinetic(const LocalSystem& ls) {
    double sum = 0.0;
    for (std::size_t i = 0; i < 3 * ls.n_owned(); ++i) sum += ls.vel[i] * ls.vel[i];
    return 0.5 * sum;
}

}  // namespace detail

// Forces on owned particles from owned and ghost neighbours (cell lists). Ghosts
// carry no forces of their own: an owned-ghost pair is computed on both ranks and
// each counts half of its energy, virial and pair count.
inline LocalSums compute_forces_local(LocalSystem& ls, double cutoff) {
    const std::size_t n_owned = ls.n_owned();
    const std::size_t n_total = ls.pos.size() / 3;
    ls.force.assign(3 * n_owned, 0.0);
    LocalSums sums;
    sums.kinetic = detail::local_kinetic(ls);
    if (n_total == 0) return sums;

    double lo[3], extent[3];
    int nc[3];
    for (int d = 0; d < 3; ++d) {
        double hi = ls.pos[d];
        lo[d] = ls.pos[d];
        for (std::size_t i = 1; i < n_total; ++i) {
            lo[d] = std::min(lo[d], ls.pos[3 * i + d]);
            hi = std::max(hi, ls.pos[3 * i + d]);
        }
        extent[d] = std::max(hi - lo[d], cutoff);
        nc[d] = std::max(1, static_cast<int>(extent[d] / cutoff));
    }
    auto cell_of = [&](std::size_t i) {
        int c[3];
        for (int d = 0; d < 3; ++d)
            c[d] = std::min(nc[d] - 1,
                            static_cast<int>((ls.pos[3 * i + d] - lo[d]) / extent[d] * nc[d]));
        return (c[0] * nc[1] + c[1]) * nc[2] + c[2];
    };

    const std::size_t n_cells = static_cast<std::size_t>(nc[0]) * nc[1] * nc[2];
    std::vector<int> start(n_cells + 1, 0), order(n_total), cell(n_total);
    for (std::size_t i = 0; i < n_total; ++i) ++start[static_cast<std::size_t>(cell[i] = cell_of(i)) + 1];
    for (std::size_t c = 0; c < n_cells; ++c) start[c + 1] += start[c];
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < n_total; ++i) order[static_cast<std::size_t>(fill[static_cast<std::size_t>(cell[i])]++)] = static_cast<int>(i);

    const double rc2 = cutoff * cutoff;
    auto interact = [&](std::size_t i, std::size_t j) {
        const bool owned_i = i < n_owned, owned_j = j < n_owned;
        if (!owned_i && !owned_j) return;
        double d[3];
        double r2 = 0.0;
        for (int k = 0; k < 3; ++k) {
            d[k] = ls.pos[3 * i + k] - ls.pos[3 * j + k];
            r2 += d[k] * d[k];
        }
        if (r2 >= rc2) return;
        const double f = lj::pair_force_over_r(r2, rc2);
        for (int k = 0; k < 3; ++k) {
            if (owned_i) ls.force[3 * i + k] += f * d[k];
            if (owned_j) ls.force[3 * j + k] -= f * d[k];
        }
        const double weight = owned_i && owned_j ? 1.0 : 0.5;
        sums.potential += weight * lj::shifted_pair_energy(r2, rc2);
        sums.virial += weight * f * r2;
        sums.pairs += weight;
    };

    for (int cx = 0; cx < nc[0]; ++cx)
        for (int cy = 0; cy < nc[1]; ++cy)
            for (int cz = 0; cz < nc[2]; ++cz) {
                const int c = (cx * nc[1] + cy) * nc[2] + cz;
                for (int nx = std::max(0, cx - 1); nx <= std::min(nc[0] - 1, cx + 1); ++nx)
                    for (int ny = std::max(0, cy - 1); ny <= std::min(nc[1] - 1, cy + 1); ++ny)
                        for (int nz = std::max(0, cz - 1); nz <= std::min(nc[2] - 1, cz + 1); ++nz) {
                            const int c2 = (nx * nc[1] + ny) * nc[2] + nz;
                            if (c2 < c) continue;
                            for (int a = start[static_cast<std::size_t>(c)]; a < start[static_cast<std::size_t>(c) + 1]; ++a)
                                for (int b = (c2 == c ? a + 1 : start[static_cast<std::size_t>(c2)]);
                                     b < start[static_cast<std::size_t>(c2) + 1]; ++b)
                                    interact(static_cast<std::size_t>(order[static_cast<std::size_t>(a)]),
                                             static_cast<std::size_t>(order[static_cast<std::size_t>(b)]));
                        }
            }
    return sums;
}

// One velocity-Verlet step: kicks, drift, migration, ghost exchange, forces.
inline LocalSums verlet_step_local(LocalSystem& ls, const Decomposition& dec, double dt,
                                   double cutoff) {
    for (std::size_t i = 0; i < 3 * ls.n_owned(); ++i) {
        ls.vel[i] += 0.5 * dt * ls.force[i];
        ls.pos[i] += dt * ls.vel[i];
    }
    migrate(ls, dec);
    exchange_ghosts(ls, dec, cutoff);
    LocalSums sums = compute_forces_local(ls, cutoff);
    for (std::size_t i = 0; i < 3 * ls.n_owned(); ++i) ls.vel[i] += 0.5 * dt * ls.force[i];
    sums.kinetic = detail::local_kinetic(ls);
    return sums;
}

}  // namespace md::mpi
