#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


#define USE_MPI 1   /* Enable or disable mpi.h usage. */


// Fox's algorithm implementation for OpenMP.
// @param threads: number of threads usage. 0 or less to use all threads. 
void fox_omp(double* A, double* B, double* C, int m_size, int threads);

// Fox's algorithm implementation for OpenMPI.
// @param threads: number of threads usage. 0 or less to use all threads. 
void fox_mpi(double* A, double* B, double* C, int m_size, int threads);


#if USE_MPI
#include <mpi.h>

// Grid.
typedef struct {
    MPI_Comm grid_comm; /* Handle to global grid communicator */
    MPI_Comm row_comm;  /* Row communicator */
    MPI_Comm col_comm;  /* Column communicator */
    int n_proc;         /* Number of processors */
    int grid_dim;       /* Dimension of the grid, = sqrt(n_proc) */
    int my_row;         /* Row position of a processor in a grid */
    int my_col;         /* Column position of a procesor in a grid */
    int my_rank;        /* Rank within the grid */
} GridInfo;


// Internal Fox algorithm implementation fo OpenMPI.
void __fox_mpi(double* A, double* B, double* C, int size, GridInfo* grid);

// Initialize grid info for OpenMPI processing.
void __grid_init(GridInfo* grid);
#endif