#include <iostream>
#include <ostream>
#include <fstream>
#include <mpi.h>

#include "mpi_utils.hpp"
#include "mpi_grid.hpp"


void fox_prod(double* A, double* B, double* C, int size, MpiGrid* grid) {
    double* buff_A = new double[size * size];
    MPI_Status status;
    int from = (grid->curr_row() + 1) % grid->grid_dim();
    int to = (grid->curr_row() - 1 + grid->grid_dim()) % grid->grid_dim();

    for (int stage = 0; stage < grid->grid_dim(); ++stage) {
        int root = (grid->curr_row() + stage) % grid->grid_dim();
        if (root == grid->curr_col()) {
            MPI_Bcast(A, size * size, MPI_DOUBLE, root, grid->row_comm());
            matr_prod(A, B, C, size);
        }
        else {
            MPI_Bcast(buff_A, size * size, MPI_DOUBLE, root, grid->row_comm());
            matr_prod(buff_A, B, C, size);
        }
        MPI_Sendrecv_replace(B, size * size, MPI_DOUBLE, to, 0, from, 0, grid->col_comm(), &status);
    }
}


int main(int argc, char** argv) {
    double* pA = nullptr;
    double* pB = nullptr;
    double* pC = nullptr;
    double* local_pA = nullptr;
    double* local_pB = nullptr;
    double* local_pC = nullptr;
    int size = 1400;

    std::ofstream os;
    os.open("output.txt", std::ios::app);
    os << "Starting...\n";
    std::cout << "Start\n";
    MPI_Init(&argc, &argv);

    MpiGrid grid;
    int local_size = grid.local_size(size);

    if (grid.is_main()) {
        pA = matr_init_ident_t(size, 4);
        pB = matr_init_ident_t(size, 2);
        pC = matr_init(size, 0);
    }

    local_pA = matr_init(local_size);
    local_pB = matr_init(local_size);
    local_pC = matr_init(local_size);

    MPI_Datatype blocktype, type;
    int array_size[2] = { size, size };
    int subarray_sizes[2] = { local_size, local_size };
    int array_start[2] = { 0, 0 };

    MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start, MPI_ORDER_C, MPI_DOUBLE, &blocktype);
    MPI_Type_create_resized(blocktype, 0, local_size * sizeof(double), &type);
    MPI_Type_commit(&type);

    int* displs = create_displacments(&grid, size);
    int* sendcounts = create_send_counts(&grid);

    MPI_Scatterv(pA, sendcounts, displs, type, local_pA, local_size * local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pB, sendcounts, displs, type, local_pB, local_size * local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = get_time(&grid);
    fox_prod(local_pA, local_pB, local_pC, local_size, &grid);
    double dt = get_time(&grid) - start_time;

    MPI_Gatherv(local_pC, local_size * local_size, MPI_DOUBLE, pC, sendcounts, displs, type, 0, MPI_COMM_WORLD);
    os << "Time: " << dt << "; size: " << size << "; proc: " << grid.n_proc() << "\n";

    if (grid.is_main()) {
        printf("Time %.5lf, Proc %d, Size %d\n", dt, grid.n_proc(), size);
        if (size <= 16) {
            matr_print(pA, size, std::cout, "A");
            matr_print(pB, size, std::cout, "\nB");
            matr_print(pC, size, std::cout, "\nC");
        }

        delete[] pA;
        delete[] pB;
        delete[] pC;
    }

    delete[] local_pA;
    delete[] local_pB;
    delete[] local_pC;
    delete[] displs;
    delete[] sendcounts;

    os.close();
    MPI_Finalize();
    return 0;
}