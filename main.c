#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "fox.h"
#include "utils.h"


#define TEST_SMOKE          1                                                   // Check if parallel algorithms are correct.
#define TEST_DEFAULT        2                                                   // Default algorithm benchmark.
#define TEST_OPEN_MP_1      4                                                   // OpenMP algorithm benchmark (1 thread).
#define TEST_OPEN_MP_4      8                                                   // OpenMP algorithm benchmark (4 threads).
#define TEST_OPEN_MP_9      16                                                  // OpenMP algorithm benchmark (9 threads).
#define TEST_OPEN_MP        TEST_OPEN_MP_1 | TEST_OPEN_MP_4 | TEST_OPEN_MP_9    // All OpenMP benchmarks.
#define TEST_OPEN_MPI_1     32                                                  // OpenMPI algorithm benchmark (1 thread).
#define TEST_OPEN_MPI_4     64                                                  // OpenMPI algorithm benchmark (4 threads).
#define TEST_OPEN_MPI_9     128                                                 // OpenMPI algorithm benchmark (9 threads).
#define TEST_OPEN_MPI       TEST_OPEN_MPI_1 | TEST_OPEN_MPI_4 | TEST_OPEN_MPI_9 // All OpenMPI benchmarks.
#define TEST_ALL            TEST_SMOKE | TEST_OPEN_MP | TEST_OPEN_MPI           // All tests.


void test_omp(int min_size, int max_size, int threads, int inc);
void test_mpi(int min_size, int max_size, int threads, int inc);
void test_seq(int min_size, int max_size, int inc);
void test_smoke(int size, int threads);

void init(double** pA, double** pB, double** pC, int size);
void clean(double* pA, double* pB, double* pC);


int main(int argc, char** argv)
{
    printf("Start\n\n");

    int type = TEST_OPEN_MP_4 | TEST_OPEN_MP_9;
    int m_size = 3000;      // matrix size.
    int size_inc = 2;       // matrix size increment for time benchmark.

#if USE_MPI
    MPI_Init(&argc, &argv);
#endif

    if (type & TEST_SMOKE)
        test_smoke(min(m_size, 100), 4);

    if (type & TEST_DEFAULT)
        test_seq(8, m_size, size_inc);

    if (type & TEST_OPEN_MP_1)
        test_omp(8, m_size, 1, size_inc);
    if (type & TEST_OPEN_MP_4)
        test_omp(8, m_size, 4, size_inc);
    if (type & TEST_OPEN_MP_9)
        test_omp(8, m_size, 9, size_inc);

    if (type & TEST_OPEN_MPI_1)
        test_mpi(8, m_size, 1, size_inc);
    if (type & TEST_OPEN_MPI_4)
        test_mpi(8, m_size, 4, size_inc);
    if (type & TEST_OPEN_MPI_9)
        test_mpi(8, m_size, 9, size_inc);

#if USE_MPI
    MPI_Finalize();
#endif

    printf("Done.\n");
    return 0;
}


void test_omp(int min_size, int max_size, int threads, int inc) {
    double* pA, * pB, * pC;
    printf("OpenMP multiplication algorithm [%d threads]\n", threads);
    printf("Size\t");
    for (int size = min_size; size <= max_size; size *= inc)
        printf("%d\t", size);

    printf("\nTime\t");
    for (int size = min_size; size <= max_size; size *= inc) {
        init(&pA, &pB, &pC, size);

        auto t0 = omp_get_wtime();
        fox_omp(pA, pB, pC, size, threads);
        printf("%.3f\t", omp_get_wtime() - t0);

        clean(pA, pB, pC);
    }

    printf("\n");
}

void test_mpi(int min_size, int max_size, int threads, int inc) {
    double* pA, * pB, * pC;
    printf("MPI Fox algorithm [%d: threads]\n", threads);

    printf("Size\t");
    for (int size = min_size; size <= max_size; size *= inc)
        printf("%d\t", size);

    printf("\nTime\t");
    for (int size = min_size; size <= max_size; size *= inc) {
        init(&pA, &pB, &pC, size);

        auto t0 = omp_get_wtime();
        fox_mpi(pA, pB, pC, size, threads);
        printf("%.3f\t", omp_get_wtime() - t0);

        clean(pA, pB, pC);
    }

    printf("\n");
}

void test_seq(int min_size, int max_size, int inc) {
    double* pA, * pB, * pC;
    printf("Default multiplication algorithm\n");
    printf("Size\t");
    for (int size = min_size; size <= max_size; size *= inc)
        printf("%d\t", size);

    printf("\nTime\t");
    for (int size = min_size; size <= max_size; size *= inc) {
        init(&pA, &pB, &pC, size);

        auto t0 = omp_get_wtime();
        matrix_prod(pA, pB, pC, size);
        printf("%.3f\t", omp_get_wtime() - t0);

        clean(pA, pB, pC);
    }

    printf("\n");
}

void test_smoke(int size, int threads) {
    double* pA, * pB, * pC,* pD, *pE;
    printf("Smoke test [%d size, %d threads]\n", size, threads);

    init(&pA, &pB, &pC, size);
    pD = matrix_fill(size, 0);
    pE = matrix_fill(size, 0);

    matrix_prod(pA, pB, pC, size);
    fox_omp(pA, pB, pD, size, threads);
    fox_mpi(pA, pB, pE, size, threads);

    if (size <= 20) {
        matrix_print(pA, size, "A");
        matrix_print(pB, size, "\nB");
        matrix_print(pC, size, "\nDefault");
        matrix_print(pD, size, "\nOpenMP");

#if USE_MPI
        matrix_print(pE, size, "\nOpenMPI");
#endif
    }

    printf("\nOpenMP equals Default =  %s\n", (matrix_equals(pC, pD, size) ? "true" : "false"));

#if USE_MPI
    printf("OpenMPI equals Default = %s\n", (matrix_equals(pC, pE, size) ? "true" : "false"));
#endif

    clean(pA, pB, pC);
    free(pD);
    free(pE);
}


void init(double** pA, double** pB, double** pC, int size) {
    (*pA) = matrix_rand(size, 2, 0);
    (*pB) = matrix_rand(size, 3, 0);
    (*pC) = matrix_fill(size, 0);
}

void clean(double* pA, double* pB, double* pC) {
    free(pA);
    free(pB);
    free(pC);
}