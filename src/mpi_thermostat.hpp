#pragma once

#include <mpi.h>

#include <cmath>

#include "mpi_exchange.hpp"

namespace md::mpi {

// Berendsen rescale with the global kinetic temperature (Allreduce over ranks).
// n_total is the global particle count; returns the temperature before rescaling.
inline double berendsen_local(LocalSystem& ls, MPI_Comm comm, std::size_t n_total,
                              double target_temp, double dt, double tau) {
    double kinetic = 0.0;
    for (std::size_t i = 0; i < 3 * ls.n_owned(); ++i) kinetic += 0.5 * ls.vel[i] * ls.vel[i];
    MPI_Allreduce(MPI_IN_PLACE, &kinetic, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double temp = 2.0 * kinetic / (3.0 * static_cast<double>(n_total - 1));
    const double lambda = std::sqrt(1.0 + dt / tau * (target_temp / temp - 1.0));
    for (double& v : ls.vel) v *= lambda;
    return temp;
}

}  // namespace md::mpi
