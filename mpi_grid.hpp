#pragma once

#include <mpi.h>
#include <math.h>


// MPI grid info object.
struct MpiGrid {
private:
    MPI_Comm _grid_comm;     // Handle to global grid communicator.
    MPI_Comm _row_comm;      // Row communicator.
    MPI_Comm _col_comm;      // Column communicator.
    int _n_proc;             // Number of processors.
    int _grid_dim;           // Dimension of the grid = sqrt(n_proc).
    int _curr_row;           // Row position of processor in a grid.
    int _curr_col;           // Column position of processor in a grid.
    int _curr_rank;          // Rank within the grid.

public:
    MpiGrid() {
        int old_rank;
        int dimensions[2];
        int wrap_around[2];
        int coordinates[2];
        int free_coords[2];

        MPI_Comm_size(MPI_COMM_WORLD, &_n_proc);
        MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

        _grid_dim = (int)sqrt(_n_proc);

        dimensions[0] = dimensions[1] = _grid_dim;
        wrap_around[0] = wrap_around[1] = 1;

        MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &_grid_comm);
        MPI_Comm_rank(_grid_comm, &_curr_rank);
        MPI_Cart_coords(_grid_comm, _curr_rank, 2, coordinates);

        _curr_row = coordinates[0];
        _curr_col = coordinates[1];

        free_coords[0] = 0;
        free_coords[1] = 1;
        MPI_Cart_sub(_grid_comm, free_coords, &_row_comm);

        free_coords[0] = 1;
        free_coords[1] = 0;
        MPI_Cart_sub(_grid_comm, free_coords, &_col_comm);
    }


    // Handle to global grid communicator.
    inline MPI_Comm grid_comm() const { return _grid_comm; }

    // Row communicator.
    inline MPI_Comm row_comm() const { return _row_comm; }

    // Column communicator.
    inline MPI_Comm col_comm() const { return _row_comm; }

    // Number of processors.
    inline int n_proc() const { return _n_proc; }

    // Dimension of the grid = sqrt(n_proc).
    inline int grid_dim() const { return _grid_dim; }

    // Row position of processor in a grid.
    inline int curr_row() const { return _curr_row; }

    // Column position of processor in a grid.
    inline int curr_col() const { return _curr_col; }

    // Rank within the grid.
    inline int curr_rank() const { return _curr_rank; }

    // Returns curr_rank == 0.
    inline bool is_main() const { return _curr_rank == 0; }

    inline int local_size(int size) const { return size / _grid_dim; }
};