#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

#include "forces.hpp"
#include "integrator.hpp"
#include "lattice.hpp"
#include "neighbor.hpp"

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

// A melted, non-trivial configuration: lattice evolved with the brute-force kernel.
md::System melted(int cells) {
    md::System s = md::make_fcc(cells, 0.8442, 0.722, 7);
    md::compute_forces(s);
    for (int i = 0; i < 200; ++i) md::verlet_step(s, 0.005);
    return s;
}

double max_force_diff(const md::System& a, const md::System& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.force.size(); ++i)
        worst = std::max(worst, std::fabs(a.force[i] - b.force[i]));
    return worst;
}

void test_matches_brute_force(const char* label, int cells) {
    md::System ref = melted(cells);
    md::System nl_sys = ref;
    const double pe_ref = md::compute_forces(ref);

    std::fill(nl_sys.force.begin(), nl_sys.force.end(), 0.0);
    md::NeighborList nl(0.3);
    nl.build(nl_sys);
    const double pe_nl = md::compute_forces(nl_sys, nl);

    std::printf("%s: N=%zu pairs=%zu\n", label, ref.n, nl.pairs().size());
    expect_near(label, pe_nl, pe_ref, 1e-9 * std::fabs(pe_ref));
    check(label, max_force_diff(ref, nl_sys) < 1e-9);

    std::set<std::pair<std::uint32_t, std::uint32_t>> unique;
    for (const auto& p : nl.pairs()) unique.insert({std::min(p.first, p.second), std::max(p.first, p.second)});
    check("neighbour pairs are unique", unique.size() == nl.pairs().size());
    check("neighbour list is not empty", !nl.pairs().empty());
}

void test_rebuild_trigger() {
    md::System s = melted(4);
    md::NeighborList nl(0.3);
    nl.build(s);
    check("no rebuild right after build", !nl.needs_rebuild(s));
    s.pos[0] += 0.10;
    check("no rebuild for displacement below skin/2", !nl.needs_rebuild(s));
    s.pos[0] += 0.10;
    check("rebuild once displacement exceeds skin/2", nl.needs_rebuild(s));
}

void test_trajectory_matches_brute_force() {
    md::System a = melted(4);
    md::System b = a;
    md::NeighborList nl(0.3);
    nl.build(b);
    double pe_a = md::compute_forces(a);
    double pe_b = md::compute_forces(b, nl);
    for (int step = 0; step < 500; ++step) {
        pe_a = md::verlet_step(a, 0.005);
        pe_b = md::verlet_step(b, 0.005, nl);
    }
    expect_near("trajectory energy matches brute force", pe_b, pe_a, 1e-8 * std::fabs(pe_a));
}

}  // namespace

int main() {
    test_matches_brute_force("2 cells per side (dedup path)", 4);
    test_matches_brute_force("3 cells per side", 6);
    test_rebuild_trigger();
    test_trajectory_matches_brute_force();
    if (failures == 0) std::puts("test_neighbor: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
