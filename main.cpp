#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "fox.hpp"
#include "matrix.hpp"


void benchamrk(int min_size, int max_size, int inc);
void test_smoke(int size, int threads);


int main(int argc, char** argv)
{
    int mode = 0;

    printf("Start\n\n");
    
    printf("Select mode (1 - smoke test, 2 - benchmark):\n> ");
    scanf_s("%d", &mode);
    
    if (mode != 1) {
        benchamrk(50, 2500, 2);
    }
    else {
        int size = 10;
        int threads = 2;

        printf("Select matrix size:\n> ");
        scanf_s("%d", &size);

        printf("Select threads count:\n> ");
        scanf_s("%d", &threads);

        test_smoke(size, threads);
    }

    printf("\nDone.\n");
    return 0;
}


void benchamrk(int min_size, int max_size, int inc) {
    printf("\nBenchmark [max threads %d]\n", omp_get_max_threads());
    
    printf("Time / Size\t");
    for (int size = min_size; size <= max_size; size *= inc)
        printf("%d\t", size);

    printf("\nDefault\t\t");
    for (int size = min_size; size <= max_size; size *= inc) {
        Matrix A(size, 123), B(size, 131);

        auto t0 = omp_get_wtime();
        Matrix* pC = prod(&A, &B);
        printf("%.3f\t", omp_get_wtime() - t0);

        delete pC;
    }

    int max = omp_get_max_threads() + 1; // можно запустить для threads = 9
    for (int i = 1; i * i <= max; i++) {
        printf("\nOMP [%d thread]\t", i*i);
        for (int size = min_size; size <= max_size; size *= inc) {
            Matrix A(size, 123), B(size, 131);

            auto t0 = omp_get_wtime();
            Matrix* pC = fox_omp(&A, &B, i*i);
            printf("%.3f\t", omp_get_wtime() - t0);

            delete pC;
        }
    }
    printf("\n");
}

void test_smoke(int size, int threads) {
    printf("\nSmoke test [%d size, %d threads]\n", size, threads);
    if (size > 20)
        printf("Matrix size %d is too much to display here...\n", size);


    Matrix A(size, 2), B(size, 3);
    if (size <= 20) {
        A.print(std::cout, "A");
        B.print(std::cout, "\nB");
    }

    auto t0 = omp_get_wtime();
    auto pC = prod(&A, &B);
    auto t1 = omp_get_wtime();

    printf("Default done in %.5f sec\n", t1 - t0);
    if (size <= 20) {
        pC->print(std::cout, "\nDefault");
    }

    auto pD = fox_omp(&A, &B, threads);
    auto t2 = omp_get_wtime();
    printf("OpenMP  done in %.5f sec\n", t2 - t1);

    if (size <= 20) {
        pD->print(std::cout, "\nOpenMP");
    }
    printf("\nOpenMP equals Default: %s\n", (pC->equals(pD) ? "true" : "false"));

    delete pC;
    delete pD;
}