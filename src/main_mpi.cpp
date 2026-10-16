// Parallel LJ MD (spatial decomposition). Berendsen equilibration, then timed NVE.
// Prints one CSV line: p,N,steps,wall_s,step_s,T,P,U,drift (rank 0).
// Usage: mpirun -np p md_mpi cells equil_steps nve_steps seed rho T rc
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "lattice.hpp"
#include "lj.hpp"
#include "mpi_decomp.hpp"
#include "mpi_exchange.hpp"
#include "mpi_forces.hpp"
#include "mpi_thermostat.hpp"
#include "thermo.hpp"

namespace {

constexpr double kDt = 0.005;
constexpr double kTau = 0.1;
constexpr int kThermoEvery = 10;

struct Global {
    double potential, virial, pairs, kinetic;
};

Global reduce(const md::mpi::LocalSums& s, MPI_Comm comm) {
    double v[4] = {s.potential, s.virial, s.pairs, s.kinetic};
    MPI_Allreduce(MPI_IN_PLACE, v, 4, MPI_DOUBLE, MPI_SUM, comm);
    return {v[0], v[1], v[2], v[3]};
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 8) {
        std::fputs("usage: md_mpi cells equil nve seed rho T rc\n", stderr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const int cells = std::atoi(argv[1]);
    const int equil = std::atoi(argv[2]);
    const int nve = std::atoi(argv[3]);
    const auto seed = static_cast<unsigned>(std::atoi(argv[4]));
    const double rho = std::atof(argv[5]);
    const double temp = std::atof(argv[6]);
    const double rc = std::atof(argv[7]);

    md::System global = md::make_fcc(cells, rho, temp, seed);
    global.cutoff = rc;
    int p = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        md::mpi::Decomposition dec(MPI_COMM_WORLD, global.box);
        md::mpi::LocalSystem ls = md::mpi::distribute(global, dec);
        md::mpi::exchange_ghosts(ls, dec, rc);
        md::mpi::compute_forces_local(ls, rc);

        for (int step = 0; step < equil; ++step) {
            md::mpi::verlet_step_local(ls, dec, kDt, rc);
            md::mpi::berendsen_local(ls, dec.comm(), global.n, temp, kDt, kTau);
        }

        MPI_Barrier(dec.comm());
        const double start = MPI_Wtime();
        md::mpi::LocalSums sums;
        Global first{}, last{};
        for (int step = 1; step <= nve; ++step) {
            sums = md::mpi::verlet_step_local(ls, dec, kDt, rc);
            if (step % kThermoEvery == 0) {
                last = reduce(sums, dec.comm());
                if (step == kThermoEvery) first = last;
            }
        }
        double wall = MPI_Wtime() - start;
        MPI_Allreduce(MPI_IN_PLACE, &wall, 1, MPI_DOUBLE, MPI_MAX, dec.comm());

        if (rank == 0) {
            const double volume = global.box * global.box * global.box;
            const double n = static_cast<double>(global.n);
            const double t_inst = 2.0 * last.kinetic / (3.0 * (n - 1.0));
            const double pressure = (2.0 * last.kinetic + last.virial) / (3.0 * volume);
            const double plain = last.potential + last.pairs * lj::energy_at_cutoff(rc * rc);
            const double e_first = first.kinetic + first.potential;
            const double drift = std::fabs((last.kinetic + last.potential - e_first) / e_first);
            std::printf("%d,%zu,%d,%.6f,%.9f,%.5f,%.5f,%.5f,%.3e\n", p, global.n, nve, wall,
                        wall / nve, t_inst, pressure, plain / n, drift);
        }
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
