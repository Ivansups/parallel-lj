#pragma once

#include <cmath>

#include "system.hpp"

namespace md {

// Berendsen weak coupling: scales velocities by sqrt(1 + dt/tau (T0/T - 1)).
inline void berendsen_rescale(System& s, double target_temp, double dt, double tau) {
    const double lambda = std::sqrt(1.0 + dt / tau * (target_temp / temperature(s) - 1.0));
    for (double& v : s.vel) v *= lambda;
}

}  // namespace md
