#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <cassert>
#include <ostream>
#include <math.h>

#include "mpi_grid.hpp"


inline int* create_displacments(MpiGrid* grid, int size) {
    int* displs = new int[grid->n_proc()];

    if (grid->curr_rank() == 0) {
        int disp = 0;
        for (int i = 0; i < grid->grid_dim(); ++i) {
            for (int j = 0; j < grid->grid_dim(); ++j) {
                displs[i * grid->grid_dim() + j] = disp;
                disp += 1;
            }
            disp += (size - 1) * grid->grid_dim();
        }
    }
    return displs;
}

inline int* create_send_counts(MpiGrid* grid) {
    int* sendcounts = new int[grid->n_proc()];

    if (grid->curr_rank() == 0) {
        for (int i = 0; i < grid->n_proc(); ++i) {
            sendcounts[i] = 1;
        }
    }
    return sendcounts;
}

inline double get_time(MpiGrid* grid) {
    MPI_Barrier(grid->grid_comm());
    if (grid->curr_rank() == 0) {
        return MPI_Wtime();
    }
    return 0;
}

inline double* matr_init(int size, double fill = 0) {
    double* matr = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        matr[i] = fill;
    }
    return matr;
}

inline double* matr_init_ident(int size, double fill) {
    double* matr = matr_init(size, 0);
    for (int i = 0; i < size; i++) {
        matr[i*size + i] = fill;
    }
    return matr;
}

inline double* matr_init_ident_t(int size, double fill) {
    double* matr = matr_init(size, 0);
    for (int i = 0; i < size; i++) {
        matr[i * size + size - i - 1] = fill;
    }
    return matr;
}

void matr_prod(double* A, double* B, double* C, int size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

bool matr_equal(double* A, double* B, int size)
{
    for (int i = 0; i < size*size; ++i) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

void matr_print(double* A, int size, std::ostream& os, const char* name = nullptr) {
    if (name != nullptr) {
        os << name << " [" << size << " x " << size << "]\n";
    }
    for (int i = 0; i < size; ++i) {
        int stride = i * size;
        for (int j = 0; j < size; ++j)
            os << A[stride + j] << "\t";
        os << "\n";
    }
}