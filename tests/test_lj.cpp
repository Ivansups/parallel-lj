#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "lj.hpp"

namespace {

int failures = 0;

void expect_near(const char* name, double actual, double expected, double tol) {
    if (std::fabs(actual - expected) > tol) {
        std::printf("FAIL %s: got %.12g, expected %.12g\n", name, actual, expected);
        ++failures;
    }
}

}  // namespace

int main() {
    const double rc2 = lj::kCutoff * lj::kCutoff;
    const double rmin = std::pow(2.0, 1.0 / 6.0);

    expect_near("U(r=1) is zero", lj::pair_energy(1.0, rc2), 0.0, 1e-12);
    expect_near("U(r=2^(1/6)) is -1", lj::pair_energy(rmin * rmin, rc2), -1.0, 1e-12);
    expect_near("F/r(r=1) is 24", lj::pair_force_over_r(1.0, rc2), 24.0, 1e-12);
    expect_near("F(r=2^(1/6)) is zero", lj::pair_force_over_r(rmin * rmin, rc2) * rmin, 0.0, 1e-12);
    expect_near("U beyond cutoff is zero", lj::pair_energy(rc2 * 1.01, rc2), 0.0, 0.0);
    expect_near("F beyond cutoff is zero", lj::pair_force_over_r(rc2 * 1.01, rc2), 0.0, 0.0);

    const double inside = rc2 * (1.0 - 1e-12);
    expect_near("shifted U is continuous at cutoff", lj::shifted_pair_energy(inside, rc2), 0.0, 1e-9);
    expect_near("shifted U equals U minus U(rc) inside",
                lj::shifted_pair_energy(1.5, rc2),
                lj::pair_energy(1.5, rc2) - lj::pair_energy(rc2 * (1.0 - 1e-15), rc2), 1e-9);
    expect_near("shifted U beyond cutoff is zero", lj::shifted_pair_energy(rc2 * 1.01, rc2), 0.0, 0.0);

    if (failures == 0) std::puts("test_lj: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
