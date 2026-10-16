#pragma once

#include <cmath>

#include "forces.hpp"
#include "system.hpp"

namespace md {

// One velocity-Verlet step (unit mass). Expects s.force to hold current forces
// and leaves it updated; returns the potential energy after the step.
inline double verlet_step(System& s, double dt) {
    for (std::size_t i = 0; i < 3 * s.n; ++i) {
        s.vel[i] += 0.5 * dt * s.force[i];
        s.pos[i] += dt * s.vel[i];
        s.pos[i] -= s.box * std::floor(s.pos[i] / s.box);
    }
    const double pe = compute_forces(s);
    for (std::size_t i = 0; i < 3 * s.n; ++i) s.vel[i] += 0.5 * dt * s.force[i];
    return pe;
}

}  // namespace md
