#pragma once

#include "lj.hpp"
#include "system.hpp"

namespace md {

inline constexpr double kPi = 3.14159265358979323846;

struct Thermo {
    double kinetic = 0.0;
    double potential = 0.0;
    double total = 0.0;
    double temperature = 0.0;
    double pressure = 0.0;
};

// Instantaneous observables; pe is the potential energy from the last force
// evaluation, whose virial is stored in s.virial. P = (2 KE + W) / (3 V).
inline Thermo measure(const System& s, double pe) {
    Thermo t;
    t.kinetic = kinetic_energy(s);
    t.potential = pe;
    t.total = t.kinetic + pe;
    t.temperature = temperature(s);
    t.pressure = (2.0 * t.kinetic + s.virial) / (3.0 * s.box * s.box * s.box);
    return t;
}

// Standard long-range corrections (g(r) = 1 beyond rc), per particle / pressure.
inline double lrc_energy_per_particle(double rho, double rc) {
    const double inv3 = 1.0 / (rc * rc * rc);
    return 8.0 / 3.0 * kPi * rho * (inv3 * inv3 * inv3 / 3.0 - inv3);
}

inline double lrc_pressure(double rho, double rc) {
    const double inv3 = 1.0 / (rc * rc * rc);
    return 16.0 / 3.0 * kPi * rho * rho * (2.0 * inv3 * inv3 * inv3 / 3.0 - inv3);
}

// Potential energy of the plain truncated potential: undoes the per-pair shift
// applied in the force kernel (pe is the shifted energy).
inline double plain_potential(const System& s, double pe) {
    return pe + static_cast<double>(s.pair_count) * lj::energy_at_cutoff(s.cutoff * s.cutoff);
}

}  // namespace md
