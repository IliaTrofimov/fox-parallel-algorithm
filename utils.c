#include "utils.h"
#include <time.h>

double* matrix_fill(size_t size, double fill)
{
    double* pA = (double*)calloc(size * size, sizeof(double));
    for (size_t i = 0; i < size * size; i++)
        pA[i] = fill;
    return pA;
}

double* matrix_rand(size_t size, int sup, size_t seed)
{
    srand(time(NULL));
    double* pA = (double*)calloc(size * size, sizeof(double));

    for (size_t i = 0; i < size * size; i++)
        pA[i] = rand() % sup + 1;

    return pA;
}

void matrix_prod(double* A, double* B, double* C, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k) {
                *(C + i * size + j) += *(A + i * size + k) * *(B + k * size + j);
            }
        }
    }
}

int matrix_equals(double* A, double* B, size_t size)
{
    for (size_t i = 0; i < size * size; ++i) {
        if (*(A + i) != *(B + i)) {
            return 0;
        }
    }
    return 1;
}

void matrix_print(double* A, size_t size, const char* name)
{
    if (name != NULL)
        printf("%s [%ld x %ld]:\n", name, size, size);

    for (size_t i = 0; i < size * size; ++i) {
        printf("%.2lf ", *(A + i));
        if ((i + 1) % size == 0) {
            printf("\n");
        }
    }
}
