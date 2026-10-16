// Serial LJ MD: Berendsen equilibration, then NVE. Thermo CSV on stdout.
// Usage: md_serial [cells=6] [equil_steps=2000] [nve_steps=2000] [seed=1]
#include <cstdio>
#include <cstdlib>

#include "integrator.hpp"
#include "lattice.hpp"
#include "neighbor.hpp"
#include "thermo.hpp"
#include "thermostat.hpp"

namespace {

constexpr double kDensity = 0.8442;
constexpr double kTemperature = 0.722;
constexpr double kDt = 0.005;
constexpr double kTau = 0.1;
constexpr int kThermoEvery = 10;

void print(const char* phase, int step, const md::Thermo& t) {
    std::printf("%s,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f\n", phase, step, step * kDt, t.kinetic,
                t.potential, t.total, t.temperature, t.pressure);
}

}  // namespace

int main(int argc, char** argv) {
    const int cells = argc > 1 ? std::atoi(argv[1]) : 6;
    const int equil = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int nve = argc > 3 ? std::atoi(argv[3]) : 2000;
    const unsigned seed = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 1u;

    md::System s = md::make_fcc(cells, kDensity, kTemperature, seed);
    md::NeighborList nl;
    nl.build(s);
    double pe = md::compute_forces(s, nl);

    std::puts("phase,step,time,kinetic,potential,total,temperature,pressure");
    for (int step = 1; step <= equil; ++step) {
        pe = md::verlet_step(s, kDt, nl);
        md::berendsen_rescale(s, kTemperature, kDt, kTau);
        if (step % kThermoEvery == 0) print("equil", step, md::measure(s, pe));
    }
    for (int step = 1; step <= nve; ++step) {
        pe = md::verlet_step(s, kDt, nl);
        if (step % kThermoEvery == 0) print("nve", step, md::measure(s, pe));
    }
    return EXIT_SUCCESS;
}
