#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

void matrix_creation(double** pA, double** pB, double** pC, int size)
{
    *pA = (double*)calloc(size * size, sizeof(double));
    *pB = (double*)calloc(size * size, sizeof(double));
    *pC = (double*)calloc(size * size, sizeof(double));
}


void matrix_initialization(double* A, double* B, int size, int sup)
{
    srand(time(NULL));
    for (int i = 0; i < size * size; ++i) {
        *(A + i) = rand() % sup + 1;
        *(B + i) = rand() % sup + 1;
    }
}


void matrix_dot(double* A, double* B, double* C, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                *(C + i * n + j) += *(A + i * n + k) * *(B + k * n + j);
            }
        }
    }
}


int matrix_check(double* A, double* B, int n)
{
    for (int i = 0; i < n * n; ++i) {
        if (*(A + i) != *(B + i)) {
            return 0;
        }
    }
    return 1;
}


void _matrix_print(double* A, int n)
{
    printf("---~---~---~---~---\n");
    for (int i = 0; i < n * n; ++i) {
        printf("%.2lf ", *(A + i));
        if ((i + 1) % n == 0) {
            printf("\n");
        }
    }
    printf("---~---~---~---~---\n");
}


void matrix_removal(double** pA, double** pB, double** pC)
{
    free(*pA);
    free(*pB);
    free(*pC);
}


/* ------------------------------ FoxAlgorithm ------------------------------ */

void FoxAlgorithm(double* A, double* B, double* C, int m_size, int threads)
{
    int stage = 0;
    int n_blocks = sqrt(threads);
    int block_size = m_size / n_blocks;

    printf("fox_omp(): m_size=%d threads=%d n_blocks=%d block_size=%d\n", m_size, threads, n_blocks, block_size);

    omp_set_num_threads(threads);
#pragma omp parallel private(stage) shared(A, B, C)
    {
        int PrNum = omp_get_thread_num();
        int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;
        printf("  [%d] fox_omp(): i1=%d j1=%d PrNum=%d\n", PrNum, i1, j1, PrNum);

        double* A1, * B1, * C1;

        for (stage = 0; stage < n_blocks; ++stage) {
            A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
            B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
            C1 = C + (i1 * m_size + j1) * block_size;
            
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    for (int k = 0; k < block_size; ++k) {
                        *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                    }
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------- */


int main1(int argc, char** argv)
{
    int m_size = 8, n_threads = 4;

    if (argc == 2) {
        sscanf_s(argv[1], "%d", &m_size);
    }

    double* pA, * pB, * pC;
    matrix_creation(&pA, &pB, &pC, m_size);
    matrix_initialization(pA, pB, m_size, 2);

    double start_time = omp_get_wtime();
    FoxAlgorithm(pA, pB, pC, m_size, n_threads);
    double end_time = omp_get_wtime() - start_time;

    printf("FoxAlgorithm_time: %.5lf n_threads: %d m_size: %d \n", end_time, n_threads, m_size);
    printf("%.5lf, %d, %d\n", end_time, n_threads, m_size);

    /* Sequential implementation comparison */
    double *pD = (double *)calloc(m_size * m_size, sizeof(double));
    double s_time = omp_get_wtime();
    matrix_dot(pA, pB, pD, m_size);
    double e_time = omp_get_wtime() - s_time;
    printf("DotAlgorithm_time: %.5lf\n", e_time);
    printf("matrix_check: %s", matrix_check(pC, pD, m_size) ? "yes" : "NO");

    _matrix_print(pC, m_size);
    _matrix_print(pD, m_size);


    free(pD);

    matrix_removal(&pA, &pB, &pC);

    return 0;
}