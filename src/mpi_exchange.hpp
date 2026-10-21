#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "mpi_decomp.hpp"
#include "system.hpp"

namespace md::mpi {

// Particles owned by this rank come first in pos; ghosts (copies of neighbours'
// particles, already shifted into this rank's frame) are appended after them.
struct LocalSystem {
    std::vector<int> id;  // global index of each owned particle
    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> force;
    std::size_t n_owned() const { return id.size(); }
};

namespace detail {

// Sends `out` to dest and returns what source sent (sizes are exchanged first).
inline std::vector<double> sendrecv(MPI_Comm comm, const std::vector<double>& out, int dest,
                                    int source) {
    int n_out = static_cast<int>(out.size()), n_in = 0;
    MPI_Sendrecv(&n_out, 1, MPI_INT, dest, 0, &n_in, 1, MPI_INT, source, 0, comm,
                 MPI_STATUS_IGNORE);
    std::vector<double> in(static_cast<std::size_t>(n_in));
    MPI_Sendrecv(out.data(), n_out, MPI_DOUBLE, dest, 1, in.data(), n_in, MPI_DOUBLE, source, 1,
                 comm, MPI_STATUS_IGNORE);
    return in;
}

// Shifts x by a box length, keeping the result inside [0, box).
inline double shift_into_box(double x, double shift, double box) {
    double y = x + shift;
    if (y >= box) y = std::nextafter(box, 0.0);
    if (y < 0.0) y = 0.0;
    return y;
}

constexpr std::size_t kMigrationRecord = 7;  // id, x, y, z, vx, vy, vz

}  // namespace detail

// Takes the particles of a global system that lie inside this rank's box.
inline LocalSystem distribute(const System& global, const Decomposition& dec) {
    LocalSystem ls;
    for (std::size_t i = 0; i < global.n; ++i) {
        bool mine = true;
        for (int d = 0; d < 3; ++d) {
            const double x = global.pos[3 * i + d];
            mine = mine && x >= dec.lo(d) && x < dec.hi(d);
        }
        if (!mine) continue;
        ls.id.push_back(static_cast<int>(i));
        for (int d = 0; d < 3; ++d) {
            ls.pos.push_back(global.pos[3 * i + d]);
            ls.vel.push_back(global.vel[3 * i + d]);
        }
    }
    ls.force.assign(3 * ls.n_owned(), 0.0);
    return ls;
}

// Sends owned particles that left the box to the neighbouring ranks (positions are
// shifted by the box length when crossing the periodic boundary).
inline void migrate(LocalSystem& ls, const Decomposition& dec) {
    ls.pos.resize(3 * ls.n_owned());  // ghosts are rebuilt afterwards
    for (int d = 0; d < 3; ++d) {
        std::vector<double> to_minus, to_plus;
        LocalSystem kept;
        for (std::size_t i = 0; i < ls.n_owned(); ++i) {
            const double x = ls.pos[3 * i + d];
            std::vector<double>* out = nullptr;
            double sent_x = x;
            if (x < dec.lo(d)) {
                out = &to_minus;
                if (dec.coords(d) == 0) sent_x = detail::shift_into_box(x, dec.box(), dec.box());
            } else if (x >= dec.hi(d)) {
                out = &to_plus;
                if (dec.coords(d) == dec.dims(d) - 1)
                    sent_x = detail::shift_into_box(x, -dec.box(), dec.box());
            }
            if (out == nullptr) {
                kept.id.push_back(ls.id[i]);
                for (int k = 0; k < 3; ++k) {
                    kept.pos.push_back(ls.pos[3 * i + k]);
                    kept.vel.push_back(ls.vel[3 * i + k]);
                }
                continue;
            }
            out->push_back(ls.id[i]);
            for (int k = 0; k < 3; ++k) out->push_back(k == d ? sent_x : ls.pos[3 * i + k]);
            for (int k = 0; k < 3; ++k) out->push_back(ls.vel[3 * i + k]);
        }
        const auto from_minus = detail::sendrecv(dec.comm(), to_plus, dec.plus_neighbor(d),
                                                 dec.minus_neighbor(d));
        const auto from_plus = detail::sendrecv(dec.comm(), to_minus, dec.minus_neighbor(d),
                                                dec.plus_neighbor(d));
        for (const auto* in : {&from_minus, &from_plus})
            for (std::size_t r = 0; r + detail::kMigrationRecord <= in->size();
                 r += detail::kMigrationRecord) {
                kept.id.push_back(static_cast<int>((*in)[r]));
                for (std::size_t k = 0; k < 3; ++k) kept.pos.push_back((*in)[r + 1 + k]);
                for (std::size_t k = 0; k < 3; ++k) kept.vel.push_back((*in)[r + 4 + k]);
            }
        ls.id = std::move(kept.id);
        ls.pos = std::move(kept.pos);
        ls.vel = std::move(kept.vel);
    }
    ls.force.assign(3 * ls.n_owned(), 0.0);
}

// Replaces ghosts with copies of every particle within `width` of this rank's box.
// Dimensions are processed in turn and later ones also forward earlier ghosts,
// which covers edges and corners. Requires width <= box length of every rank.
inline void exchange_ghosts(LocalSystem& ls, const Decomposition& dec, double width) {
    ls.pos.resize(3 * ls.n_owned());
    for (int d = 0; d < 3; ++d) {
        std::vector<double> to_minus, to_plus;
        const std::size_t n = ls.pos.size() / 3;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = ls.pos[3 * i + d];
            if (x < dec.lo(d) + width) {
                for (int k = 0; k < 3; ++k)
                    to_minus.push_back(k == d && dec.coords(d) == 0 ? x + dec.box() : ls.pos[3 * i + k]);
            }
            if (x >= dec.hi(d) - width) {
                for (int k = 0; k < 3; ++k)
                    to_plus.push_back(k == d && dec.coords(d) == dec.dims(d) - 1
                                          ? x - dec.box()
                                          : ls.pos[3 * i + k]);
            }
        }
        const auto from_plus = detail::sendrecv(dec.comm(), to_minus, dec.minus_neighbor(d),
                                                dec.plus_neighbor(d));
        const auto from_minus = detail::sendrecv(dec.comm(), to_plus, dec.plus_neighbor(d),
                                                 dec.minus_neighbor(d));
        ls.pos.insert(ls.pos.end(), from_plus.begin(), from_plus.end());
        ls.pos.insert(ls.pos.end(), from_minus.begin(), from_minus.end());
    }
}

}  // namespace md::mpi
