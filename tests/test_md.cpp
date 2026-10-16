#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "forces.hpp"
#include "integrator.hpp"
#include "lattice.hpp"
#include "lj.hpp"
#include "system.hpp"

namespace {

int failures = 0;

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

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

void test_min_image() {
    expect_near("min_image wraps +", md::min_image(4.0, 5.0), -1.0, 1e-12);
    expect_near("min_image wraps -", md::min_image(-4.0, 5.0), 1.0, 1e-12);
    expect_near("min_image keeps small", md::min_image(1.5, 5.0), 1.5, 1e-12);
}

void test_lattice() {
    const double rho = 0.8442, temp = 0.722;
    md::System s = md::make_fcc(3, rho, temp, 42);
    check("fcc particle count is 4*cells^3", s.n == 108);
    expect_near("box matches density", static_cast<double>(s.n) / (s.box * s.box * s.box),
                rho, 1e-12);

    bool inside = s.pos.size() == 3 * s.n;
    for (double x : s.pos) inside = inside && x >= 0.0 && x < s.box;
    check("positions inside the box", inside);

    double rmin2 = 1e30;
    for (std::size_t i = 0; i < s.n; ++i)
        for (std::size_t j = i + 1; j < s.n; ++j) {
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double d = md::min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
                r2 += d * d;
            }
            rmin2 = std::min(rmin2, r2);
        }
    const double a = s.box / 3.0;
    expect_near("nearest neighbour is a/sqrt(2)", std::sqrt(rmin2), a / std::sqrt(2.0), 1e-9);

    double p[3] = {0, 0, 0};
    for (std::size_t i = 0; i < s.n; ++i)
        for (int k = 0; k < 3; ++k) p[k] += s.vel[3 * i + k];
    expect_near("zero total momentum", std::fabs(p[0]) + std::fabs(p[1]) + std::fabs(p[2]), 0.0,
                1e-10);
    expect_near("initial temperature", md::temperature(s), temp, 1e-10);

    md::System same = md::make_fcc(3, rho, temp, 42);
    check("same seed gives same velocities", same.vel == s.vel);
}

void test_forces_two_particles() {
    const double r = 1.2, box = 10.0;
    md::System s = two_particles(r, box);
    const double pe = md::compute_forces(s);
    expect_near("two-particle energy", pe,
                lj::shifted_pair_energy(r * r, lj::kCutoff * lj::kCutoff), 1e-12);
    const double fx = lj::pair_force_over_r(r * r, lj::kCutoff * lj::kCutoff) * r;
    expect_near("force on particle 1 (attractive pulls toward 0)", s.force[3], fx, 1e-12);
    expect_near("Newton third law", s.force[0] + s.force[3], 0.0, 1e-12);

    const double h = 1e-6;
    md::System plus = two_particles(r + h, box), minus = two_particles(r - h, box);
    const double numeric = -(md::compute_forces(plus) - md::compute_forces(minus)) / (2 * h);
    expect_near("force is minus gradient of energy", s.force[3], numeric, 1e-6);
}

void test_forces_minimum_image() {
    md::System s = two_particles(0.0, 10.0);
    s.pos = {0.5, 5.0, 5.0, 9.5, 5.0, 5.0};  // 1.0 apart through the periodic boundary
    const double pe = md::compute_forces(s);
    expect_near("energy through boundary", pe, lj::shifted_pair_energy(1.0, 6.25), 1e-12);
}

void test_nve_energy_drift() {
    md::System s = md::make_fcc(4, 0.8442, 0.722, 12345);
    const double dt = 0.005;
    double pe = md::compute_forces(s);
    const double e0 = md::kinetic_energy(s) + pe;
    check("initial energy is finite and negative", std::isfinite(e0) && e0 < 0.0);
    double worst = 0.0;
    for (int step = 0; step < 1000; ++step) {
        pe = md::verlet_step(s, dt);
        const double e = md::kinetic_energy(s) + pe;
        worst = std::max(worst, std::fabs((e - e0) / e0));
    }
    std::printf("NVE max relative energy deviation over 1000 steps: %.3e\n", worst);
    check("NVE energy drift below 1e-4", worst < 1e-4);
}

}  // namespace

int main() {
    test_min_image();
    test_lattice();
    test_forces_two_particles();
    test_forces_minimum_image();
    test_nve_energy_drift();
    if (failures == 0) std::puts("test_md: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
