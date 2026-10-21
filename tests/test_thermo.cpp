#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "forces.hpp"
#include "integrator.hpp"
#include "lattice.hpp"
#include "lj.hpp"
#include "thermo.hpp"
#include "thermostat.hpp"

namespace {

int failures = 0;

void expect_near(const char* name, double actual, double expected, double tol) {
    if (!(std::fabs(actual - expected) <= tol)) {
        std::printf("FAIL %s: got %.12g, expected %.12g\n", name, actual, expected);
        ++failures;
    }
}

md::System two_particles(double r, double box) {
    md::System s;
    s.n = 2;
    s.box = box;
    s.pos = {1.0, 1.0, 1.0, 1.0 + r, 1.0, 1.0};
    s.vel.assign(6, 0.0);
    s.force.assign(6, 0.0);
    return s;
}

void test_virial_two_particles() {
    const double r = 1.2;
    md::System s = two_particles(r, 10.0);
    md::compute_forces(s);
    const double rc2 = lj::kCutoff * lj::kCutoff;
    expect_near("virial of one pair is f/r * r^2", s.virial, lj::pair_force_over_r(r * r, rc2) * r * r,
                1e-12);
}

// P = -dU/dV for a uniform rescaling of positions and box (zero kinetic energy).
void test_pressure_is_volume_derivative() {
    md::System s = md::make_fcc(4, 0.8442, 0.0 + 0.722, 3);
    std::fill(s.vel.begin(), s.vel.end(), 0.0);
    const double pe = md::compute_forces(s);
    const double p = md::measure(s, pe).pressure;

    const double h = 1e-5;
    auto energy_at = [&](double scale) {
        md::System t = s;
        t.box *= scale;
        for (double& x : t.pos) x *= scale;
        return md::compute_forces(t);
    };
    const double v = s.box * s.box * s.box;
    const double dv = v * (std::pow(1 + h, 3) - std::pow(1 - h, 3));
    const double numeric = -(energy_at(1 + h) - energy_at(1 - h)) / dv;
    expect_near("pressure equals minus dU/dV", p, numeric, 1e-6 * std::fabs(numeric) + 1e-6);
}

void test_ideal_gas_pressure() {
    md::System s = two_particles(5.0, 20.0);  // beyond the cutoff: no interaction
    s.vel = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
    const double pe = md::compute_forces(s);
    const md::Thermo t = md::measure(s, pe);
    expect_near("kinetic energy", t.kinetic, 1.0, 1e-12);
    expect_near("potential energy", t.potential, 0.0, 1e-12);
    expect_near("total energy", t.total, 1.0, 1e-12);
    expect_near("ideal gas pressure 2KE/(3V)", t.pressure, 2.0 / (3.0 * 8000.0), 1e-12);
    expect_near("temperature", t.temperature, md::temperature(s), 1e-12);
}

void test_berendsen() {
    const double target = 0.722, dt = 0.005;
    md::System s = md::make_fcc(3, 0.8442, 1.5, 11);

    md::System strong = s;
    md::berendsen_rescale(strong, target, dt, dt);  // tau == dt: exact rescale
    expect_near("tau = dt reaches the target exactly", md::temperature(strong), target, 1e-10);

    md::System weak = s;
    md::berendsen_rescale(weak, target, dt, 100.0);
    const double t_weak = md::temperature(weak);
    if (!(t_weak < 1.5 - 1e-5 && t_weak > 1.5 - 1e-3)) {
        std::printf("FAIL weak coupling moves toward target: T=%g\n", t_weak);
        ++failures;
    }

    md::System at_target = md::make_fcc(3, 0.8442, target, 11);
    md::System copy = at_target;
    md::berendsen_rescale(copy, target, dt, 0.1);
    expect_near("no change at target", md::temperature(copy), target, 1e-12);
}

}  // namespace

int main() {
    test_virial_two_particles();
    test_pressure_is_volume_derivative();
    test_ideal_gas_pressure();
    test_berendsen();
    if (failures == 0) std::puts("test_thermo: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
