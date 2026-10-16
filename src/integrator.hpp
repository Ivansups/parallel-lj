#pragma once

#include <cmath>

#include "forces.hpp"
#include "neighbor.hpp"
#include "system.hpp"

namespace md {

namespace detail {

// Velocity Verlet (unit mass); recompute(s) must refresh s.force and return the energy.
template <class ForceFn>
double verlet_step(System& s, double dt, ForceFn recompute) {
    for (std::size_t i = 0; i < 3 * s.n; ++i) {
        s.vel[i] += 0.5 * dt * s.force[i];
        s.pos[i] += dt * s.vel[i];
        s.pos[i] -= s.box * std::floor(s.pos[i] / s.box);
    }
    const double pe = recompute(s);
    for (std::size_t i = 0; i < 3 * s.n; ++i) s.vel[i] += 0.5 * dt * s.force[i];
    return pe;
}

}  // namespace detail

// One step with all-pairs forces. Expects s.force to hold current forces;
// returns the potential energy after the step.
inline double verlet_step(System& s, double dt) {
    return detail::verlet_step(s, dt, [](System& sys) { return compute_forces(sys); });
}

// Same step, forces from the neighbour list (rebuilt when particles moved too far).
inline double verlet_step(System& s, double dt, NeighborList& nl) {
    return detail::verlet_step(s, dt, [&nl](System& sys) {
        if (nl.needs_rebuild(sys)) nl.build(sys);
        return compute_forces(sys, nl);
    });
}

}  // namespace md
