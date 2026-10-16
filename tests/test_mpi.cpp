// Run under mpirun with p = 1, 2, 4, 8. Every rank holds the global reference system.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "forces.hpp"
#include "integrator.hpp"
#include "lattice.hpp"
#include "mpi_decomp.hpp"
#include "mpi_exchange.hpp"
#include "mpi_forces.hpp"
#include "mpi_thermostat.hpp"
#include "thermostat.hpp"

namespace {

int failures = 0;
int my_rank = 0;

void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("[rank %d] FAIL %s\n", my_rank, name);
        ++failures;
    }
}

void expect_near(const char* name, double actual, double expected, double tol) {
    if (!(std::fabs(actual - expected) <= tol)) {
        std::printf("[rank %d] FAIL %s: got %.12g, expected %.12g\n", my_rank, name, actual, expected);
        ++failures;
    }
}

double sum_all(double x) {
    double total = 0.0;
    MPI_Allreduce(&x, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Melted reference configuration, identical on every rank.
md::System reference() {
    md::System s = md::make_fcc(4, 0.8442, 0.722, 7);  // N = 256, box = 6.72
    md::compute_forces(s);
    for (int i = 0; i < 100; ++i) md::verlet_step(s, 0.005);
    for (double& x : s.pos) x -= s.box * std::floor(x / s.box);
    return s;
}

void test_decomposition(const md::mpi::Decomposition& dec) {
    int grid = 1;
    for (int d = 0; d < 3; ++d) grid *= dec.dims(d);
    check("process grid uses all ranks", grid == dec.size());

    double volume = 1.0;
    for (int d = 0; d < 3; ++d) volume *= dec.hi(d) - dec.lo(d);
    expect_near("boxes tile the cell", sum_all(volume), dec.box() * dec.box() * dec.box(), 1e-9);

    for (int d = 0; d < 3; ++d) {
        // What my minus neighbour sends to its plus neighbour (me) is its own rank.
        int received = -1;
        const int me = dec.rank();
        MPI_Sendrecv(&me, 1, MPI_INT, dec.plus_neighbor(d), 0, &received, 1, MPI_INT,
                     dec.minus_neighbor(d), 0, dec.comm(), MPI_STATUS_IGNORE);
        check("minus and plus neighbours are consistent", received == dec.minus_neighbor(d));
    }
}

void test_distribute(const md::System& global, const md::mpi::Decomposition& dec) {
    const md::mpi::LocalSystem ls = md::mpi::distribute(global, dec);
    expect_near("every particle has exactly one owner", sum_all(static_cast<double>(ls.n_owned())),
                static_cast<double>(global.n), 0.0);
    bool inside = true;
    for (std::size_t i = 0; i < ls.n_owned(); ++i)
        for (int d = 0; d < 3; ++d) {
            const double x = ls.pos[3 * i + d];
            inside = inside && x >= dec.lo(d) && x < dec.hi(d);
        }
    check("owned particles lie inside the box", inside);
}

// Deterministic displacement of particle `id`, independent of the rank that owns it.
double displacement(int id, int d) { return 0.35 * std::sin(1.7 * id + 2.3 * d + 0.5); }

void test_migration(const md::System& global, const md::mpi::Decomposition& dec) {
    md::mpi::LocalSystem ls = md::mpi::distribute(global, dec);
    for (std::size_t i = 0; i < ls.n_owned(); ++i)
        for (int d = 0; d < 3; ++d) ls.pos[3 * i + d] += displacement(ls.id[i], d);
    md::mpi::migrate(ls, dec);

    expect_near("migration conserves the particle count", sum_all(static_cast<double>(ls.n_owned())),
                static_cast<double>(global.n), 0.0);
    const double n = static_cast<double>(global.n);
    double id_sum = 0.0, id_sq = 0.0;
    bool inside = true, positions_ok = true;
    for (std::size_t i = 0; i < ls.n_owned(); ++i) {
        id_sum += ls.id[i];
        id_sq += static_cast<double>(ls.id[i]) * ls.id[i];
        for (int d = 0; d < 3; ++d) {
            const double x = ls.pos[3 * i + d];
            inside = inside && x >= dec.lo(d) && x < dec.hi(d);
            double want = global.pos[3 * static_cast<std::size_t>(ls.id[i]) + d] + displacement(ls.id[i], d);
            want -= global.box * std::floor(want / global.box);
            positions_ok = positions_ok && std::fabs(x - want) < 1e-9;
        }
    }
    check("migrated particles lie inside the box", inside);
    check("migrated positions are the wrapped moved positions", positions_ok);
    expect_near("every id survives migration (sum)", sum_all(id_sum), n * (n - 1) / 2, 0.0);
    expect_near("every id survives migration (sum of squares)", sum_all(id_sq),
                (n - 1) * n * (2 * n - 1) / 6, 0.0);
}

void test_forces_match_serial(md::System global, const md::mpi::Decomposition& dec) {
    const double pe_ref = md::compute_forces(global);
    md::mpi::LocalSystem ls = md::mpi::distribute(global, dec);
    md::mpi::exchange_ghosts(ls, dec, global.cutoff);
    const md::mpi::LocalSums sums = md::mpi::compute_forces_local(ls, global.cutoff);

    expect_near("MPI potential energy equals serial", sum_all(sums.potential), pe_ref,
                1e-9 * std::fabs(pe_ref));
    expect_near("MPI virial equals serial", sum_all(sums.virial), global.virial,
                1e-9 * std::fabs(global.virial));
    expect_near("MPI pair count equals serial", sum_all(sums.pairs),
                static_cast<double>(global.pair_count), 1e-9);
    double worst = 0.0;
    for (std::size_t i = 0; i < ls.n_owned(); ++i)
        for (int d = 0; d < 3; ++d)
            worst = std::max(worst, std::fabs(ls.force[3 * i + d] -
                                              global.force[3 * static_cast<std::size_t>(ls.id[i]) + d]));
    double global_worst = 0.0;
    MPI_Allreduce(&worst, &global_worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    check("MPI forces equal serial forces", global_worst < 1e-9);
}

void test_trajectory_matches_serial(md::System serial, const md::mpi::Decomposition& dec) {
    md::mpi::LocalSystem ls = md::mpi::distribute(serial, dec);
    md::mpi::exchange_ghosts(ls, dec, serial.cutoff);
    md::mpi::compute_forces_local(ls, serial.cutoff);
    md::compute_forces(serial);

    double pe_serial = 0.0;
    md::mpi::LocalSums sums;
    for (int step = 0; step < 100; ++step) {
        pe_serial = md::verlet_step(serial, 0.005);
        sums = md::mpi::verlet_step_local(ls, dec, 0.005, serial.cutoff);
    }
    const double e_serial = md::kinetic_energy(serial) + pe_serial;
    const double e_mpi = sum_all(sums.kinetic + sums.potential);
    expect_near("100-step trajectory energy equals serial", e_mpi, e_serial, 1e-8 * std::fabs(e_serial));
    expect_near("particles survive 100 steps", sum_all(static_cast<double>(ls.n_owned())),
                static_cast<double>(serial.n), 0.0);
}

void test_berendsen_matches_serial(md::System serial, const md::mpi::Decomposition& dec) {
    md::mpi::LocalSystem ls = md::mpi::distribute(serial, dec);
    const double before = md::temperature(serial);
    const double target = 0.5, dt = 0.005, tau = 0.1;
    const double t_reported = md::mpi::berendsen_local(ls, dec.comm(), serial.n, target, dt, tau);
    md::berendsen_rescale(serial, target, dt, tau);

    expect_near("global temperature before rescale", t_reported, before, 1e-10);
    double local_ke = 0.0;
    for (std::size_t i = 0; i < 3 * ls.n_owned(); ++i) local_ke += 0.5 * ls.vel[i] * ls.vel[i];
    expect_near("kinetic energy after rescale equals serial", sum_all(local_ke),
                md::kinetic_energy(serial), 1e-9 * md::kinetic_energy(serial));
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const md::System global = reference();
    {
        md::mpi::Decomposition dec(MPI_COMM_WORLD, global.box);
        test_decomposition(dec);
        test_distribute(global, dec);
        test_migration(global, dec);
        test_forces_match_serial(global, dec);
        test_trajectory_matches_serial(global, dec);
        test_berendsen_matches_serial(global, dec);
    }

    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0 && total_failures == 0) std::puts("test_mpi: all passed");
    MPI_Finalize();
    return total_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
