#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "forces.hpp"
#include "integrator.hpp"
#include "lattice.hpp"
#include "lj.hpp"
#include "msd.hpp"
#include "neighbor.hpp"
#include "rdf.hpp"
#include "thermo.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
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

void test_cutoff_is_a_parameter() {
    md::System s = two_particles(2.8, 10.0);
    expect_near("default cutoff 2.5 ignores r=2.8", md::compute_forces(s), 0.0, 0.0);
    s.cutoff = 3.0;
    expect_near("cutoff 3.0 sees r=2.8", md::compute_forces(s),
                lj::shifted_pair_energy(2.8 * 2.8, 9.0), 1e-12);
}

void test_neighbor_list_with_cutoff_3() {
    md::System ref = md::make_fcc(5, 0.84, 0.85, 5);
    ref.cutoff = 3.0;
    md::compute_forces(ref);
    for (int i = 0; i < 100; ++i) md::verlet_step(ref, 0.005);
    md::System nl_sys = ref;
    std::fill(nl_sys.force.begin(), nl_sys.force.end(), 0.0);
    const double pe_ref = md::compute_forces(ref);
    md::NeighborList nl(0.3);
    nl.build(nl_sys);
    const double pe_nl = md::compute_forces(nl_sys, nl);
    expect_near("neighbour list energy at rc=3", pe_nl, pe_ref, 1e-9 * std::fabs(pe_ref));
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.force.size(); ++i)
        worst = std::max(worst, std::fabs(ref.force[i] - nl_sys.force[i]));
    check("neighbour list forces at rc=3", worst < 1e-9);
    check("pair counts agree", ref.pair_count == nl_sys.pair_count && ref.pair_count > 0);
}

void test_plain_potential() {
    md::System s = md::make_fcc(4, 0.8442, 0.722, 9);
    md::compute_forces(s);
    for (int i = 0; i < 100; ++i) md::verlet_step(s, 0.005);
    const double pe = md::compute_forces(s);

    const double rc2 = s.cutoff * s.cutoff;
    double plain = 0.0;
    for (std::size_t i = 0; i < s.n; ++i)
        for (std::size_t j = i + 1; j < s.n; ++j) {
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double d = md::min_image(s.pos[3 * i + k] - s.pos[3 * j + k], s.box);
                r2 += d * d;
            }
            plain += lj::pair_energy(r2, rc2);
        }
    expect_near("plain truncated energy", md::plain_potential(s, pe), plain, 1e-9 * std::fabs(plain));
}

void test_long_range_corrections() {
    // Values from the closed forms U/N = (8/3) pi rho [rc^-9/3 - rc^-3],
    // P = (16/3) pi rho^2 [2 rc^-9/3 - rc^-3], evaluated independently in Python.
    expect_near("LRC energy rho=0.84 rc=3", md::lrc_energy_per_particle(0.84, 3.0), -0.2605166599, 1e-9);
    expect_near("LRC pressure rho=0.84 rc=3", md::lrc_pressure(0.84, 3.0), -0.4374677746, 1e-9);
    expect_near("LRC energy rho=0.8442 rc=2.5", md::lrc_energy_per_particle(0.8442, 2.5), -0.4520126248, 1e-9);
    expect_near("LRC pressure rho=0.8442 rc=2.5", md::lrc_pressure(0.8442, 2.5), -0.7621346985, 1e-9);
}

double shell_volume(const md::Rdf& rdf, int bin) {
    const double lo = rdf.r(bin) - 0.5 * rdf.dr(), hi = rdf.r(bin) + 0.5 * rdf.dr();
    return 4.0 * kPi / 3.0 * (hi * hi * hi - lo * lo * lo);
}

void test_rdf_ideal_gas() {
    md::System s;
    s.n = 2000;
    s.box = 12.0;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0.0, s.box);
    s.pos.resize(3 * s.n);
    for (double& x : s.pos) x = u(rng);
    s.vel.assign(3 * s.n, 0.0);
    s.force.assign(3 * s.n, 0.0);

    md::Rdf rdf(12, 6.0);
    rdf.accumulate(s);
    const std::vector<double> g = rdf.g();
    check("rdf has the requested bins", g.size() == 12);
    double worst = 0.0;
    for (double v : g) worst = std::max(worst, std::fabs(v - 1.0));
    std::printf("ideal gas max |g-1| = %.3f\n", worst);
    check("ideal gas g(r) is 1 within noise", !g.empty() && worst < 0.15);
}

void test_rdf_fcc_coordination() {
    md::System s = md::make_fcc(4, 0.8442, 0.722, 1);
    md::Rdf rdf(100, s.box / 2.0);
    rdf.accumulate(s);
    const std::vector<double> g = rdf.g();
    const double rho = static_cast<double>(s.n) / (s.box * s.box * s.box);
    double neighbours = 0.0;
    for (std::size_t k = 0; k < g.size(); ++k) {
        const int bin = static_cast<int>(k);
        if (rdf.r(bin) + 0.5 * rdf.dr() <= 1.4)
            neighbours += g[k] * rho * shell_volume(rdf, bin);
    }
    expect_near("FCC first shell has 12 neighbours", neighbours, 12.0, 1e-9);
}

void test_msd_free_flight() {
    md::System s;
    s.n = 2;
    s.box = 5.0;
    s.pos = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    s.vel = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0};
    s.force.assign(6, 0.0);
    md::MsdTracker tracker(s);
    expect_near("msd starts at zero", tracker.msd(), 0.0, 1e-12);
    const double dt = 0.1;
    for (int step = 0; step < 100; ++step) {
        for (std::size_t i = 0; i < 6; ++i) {
            s.pos[i] += dt * s.vel[i];
            s.pos[i] -= s.box * std::floor(s.pos[i] / s.box);
        }
        tracker.update(s);
    }
    // t = 10: distances 10 and 20, far beyond the box, so unwrapping is required.
    expect_near("msd unwraps periodic motion", tracker.msd(), (100.0 + 400.0) / 2.0, 1e-8);
}

}  // namespace

int main() {
    test_cutoff_is_a_parameter();
    test_neighbor_list_with_cutoff_3();
    test_plain_potential();
    test_long_range_corrections();
    test_rdf_ideal_gas();
    test_rdf_fcc_coordination();
    test_msd_free_flight();
    if (failures == 0) std::puts("test_validation: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
