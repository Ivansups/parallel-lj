#pragma once

#include <mpi.h>

namespace md::mpi {

// 3D periodic process grid; each rank owns the box [lo, hi) in every dimension.
class Decomposition {
public:
    Decomposition(MPI_Comm world, double box) : box_(box) {
        MPI_Comm_size(world, &size_);
        MPI_Dims_create(size_, 3, dims_);
        const int periods[3] = {1, 1, 1};
        MPI_Cart_create(world, 3, dims_, periods, 0, &cart_);
        MPI_Comm_rank(cart_, &rank_);
        MPI_Cart_coords(cart_, rank_, 3, coords_);
        for (int d = 0; d < 3; ++d) {
            lo_[d] = coords_[d] == 0 ? 0.0 : box * coords_[d] / dims_[d];
            hi_[d] = coords_[d] == dims_[d] - 1 ? box : box * (coords_[d] + 1) / dims_[d];
            MPI_Cart_shift(cart_, d, 1, &minus_[d], &plus_[d]);
        }
    }
    ~Decomposition() { MPI_Comm_free(&cart_); }
    Decomposition(const Decomposition&) = delete;
    Decomposition& operator=(const Decomposition&) = delete;

    MPI_Comm comm() const { return cart_; }
    int rank() const { return rank_; }
    int size() const { return size_; }
    double box() const { return box_; }
    int dims(int d) const { return dims_[d]; }
    int coords(int d) const { return coords_[d]; }
    double lo(int d) const { return lo_[d]; }
    double hi(int d) const { return hi_[d]; }
    int minus_neighbor(int d) const { return minus_[d]; }
    int plus_neighbor(int d) const { return plus_[d]; }

private:
    double box_;
    MPI_Comm cart_ = MPI_COMM_NULL;
    int size_ = 1, rank_ = 0;
    int dims_[3] = {0, 0, 0};
    int coords_[3] = {0, 0, 0};
    double lo_[3] = {0, 0, 0}, hi_[3] = {0, 0, 0};
    int minus_[3] = {0, 0, 0}, plus_[3] = {0, 0, 0};
};

}  // namespace md::mpi
