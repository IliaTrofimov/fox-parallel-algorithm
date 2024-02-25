#pragma once

#include <omp.h>
#include "fox.h"
#include "utils.h"


void fox_omp(double* A, double* B, double* C, int m_size, int threads)
{
    int stage = 0;
    int n_blocks = sqrt(threads);
    int block_size = m_size / n_blocks;

    if (threads <= 0)
        threads = omp_get_max_threads();

    omp_set_num_threads(threads);

#pragma omp parallel private(stage) shared(A, B, C)
    {
        int PrNum = omp_get_thread_num();
        int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;

        double* A1, * B1, * C1;
        
        for (stage = 0; stage < n_blocks; ++stage) {
            A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
            B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
            C1 = C + (i1 * m_size + j1) * block_size;

            for (int i = 0; i < block_size; ++i) {
                double* C1_i_stride = C1 + i * m_size;  // precompute all strides
                double* A1_i_stride = A1 + i * m_size;

                for (int j = 0; j < block_size; ++j) {
                    double* B1_j_stride = B1 + j;

                    for (int k = 0; k < block_size; ++k) {
                        *(C1_i_stride + j) += *(A1_i_stride + k) * *(B1_j_stride + k * m_size);
                    }
                }
            }
        }
    }
}

void fox_mpi(double* A, double* B, double* C, int m_size, int threads) {
#if USE_MPI
    GridInfo grid;
    __grid_init(&grid);

    /* ERROR check: */
    if (m_size % grid.grid_dim != 0) {
        printf("[!] matrix_size mod sqrt(n_processes) != 0 !\n");
        exit(-1);
    }

    int local_matrix_size = m_size / grid.grid_dim;
    double* local_pA = matrix_fill(local_matrix_size, 0);
    double* local_pB = matrix_fill(local_matrix_size, 0);
    double* local_pC = matrix_fill(local_matrix_size, 0);

    MPI_Datatype blocktype, type;
    int array_size[2] = { m_size, m_size };
    int subarray_sizes[2] = { local_matrix_size, local_matrix_size };
    int array_start[2] = { 0, 0 };


    MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start,
        MPI_ORDER_C, MPI_DOUBLE, &blocktype);
    MPI_Type_create_resized(blocktype, 0, local_matrix_size * sizeof(double), &type);
    MPI_Type_commit(&type);

    int* displs = (int*)malloc(grid.n_proc * sizeof(int));
    int* sendcounts = (int*)malloc(grid.n_proc * sizeof(int));

    if (grid.my_rank == 0) {
        for (int i = 0; i < grid.n_proc; ++i) {
            sendcounts[i] = 1;
        }
        int disp = 0;
        for (int i = 0; i < grid.grid_dim; ++i) {
            for (int j = 0; j < grid.grid_dim; ++j) {
                displs[i * grid.grid_dim + j] = disp;
                disp += 1;
            }
            disp += (local_matrix_size - 1) * grid.grid_dim;
        }
    }

    MPI_Scatterv(A, sendcounts, displs, type, local_pA,
        local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, type, local_pB,
        local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time, end_time;
    MPI_Barrier(grid.grid_comm);
    if (grid.my_rank == 0) {
        start_time = MPI_Wtime();
    }

    __fox_mpi(local_pA, local_pB, local_pC, local_matrix_size, &grid);

    MPI_Barrier(grid.grid_comm);
    if (grid.my_rank == 0)
        end_time = MPI_Wtime() - start_time;

    MPI_Gatherv(local_pC, local_matrix_size * local_matrix_size, MPI_DOUBLE, C, sendcounts, displs, type, 0, MPI_COMM_WORLD);

    free(local_pA); free(local_pB); free(local_pC);
    free(displs, sendcounts);
#endif
}


#if USE_MPI
#include <mpi.h>

void __grid_init(GridInfo* grid)
{
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* get the overall information before overlaying cart_grid */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->n_proc));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    printf("MPI_Comm_size(MPI_COMM_WORLD) = %d, MPI_Comm_rank(MPI_COMM_WORLD) = %d\n", grid->n_proc, old_rank);

    grid->grid_dim = (int)sqrt(grid->n_proc);

    /* ERROR check: */
    if (grid->grid_dim * grid->grid_dim != grid->n_proc) {
        printf("[!] \'-np\' is a perfect square!\n");
        exit(-1);
    }
    /* set the dimensions */
    dimensions[0] = dimensions[1] = grid->grid_dim;
    wrap_around[0] = wrap_around[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->grid_comm));

    /* since we have set reorder to true, this might have changed the ranks */
    MPI_Comm_rank(grid->grid_comm, &(grid->my_rank));

    /* get the cartesian coordinates for the current process */
    MPI_Cart_coords(grid->grid_comm, grid->my_rank, 2, coordinates);

    /* set the coordinate values for the current process */
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* create row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

    /* create column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));
}

void __fox_mpi(double* A, double* B, double* C, int size, GridInfo* grid)
{
    double* buff_A = (double*)calloc(size * size, sizeof(double));
    MPI_Status status;
    int root;
    int src = (grid->my_row + 1) % grid->grid_dim;
    int dst = (grid->my_row - 1 + grid->grid_dim) % grid->grid_dim;

    /**
     * For each iterations:
     *   1. find the blocks that are forming a diagonal
     *   2. shared that block on the row it belongs to
     *   3. multiply the updated A (or buff_A) with B onto C
     *   4. shift the B blocks upward
     */
    for (int stage = 0; stage < grid->grid_dim; ++stage) {
        root = (grid->my_row + stage) % grid->grid_dim;
        if (root == grid->my_col) {
            MPI_Bcast(A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_prod(A, B, C, size);
        }
        else {
            MPI_Bcast(buff_A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_prod(buff_A, B, C, size);
        }
        MPI_Sendrecv_replace(B, size * size, MPI_DOUBLE, dst, 0, src, 0, grid->col_comm, &status);
    }
}
#endif