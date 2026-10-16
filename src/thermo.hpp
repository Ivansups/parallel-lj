#pragma once

#include "system.hpp"

namespace md {

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

}  // namespace md
