#pragma once

#include <omp.h>
#include "matrix.hpp"


// Fox's algorithm implementation for OpenMP.
// @param A: pointer to left matrix
// @param B: pointer to right matrix
// @param threads: number of threads to use
Matrix* fox_omp(Matrix* A, Matrix* B, int threads) {
    if (A->size() != B->size())
        return nullptr;

    size_t size = A->size();
    size_t n_blocks = sqrt(threads);
    size_t block_size = size / n_blocks;

    Matrix* C = new Matrix(size);
    omp_set_num_threads(threads);

    #pragma omp parallel nowait
    {
        int threadId = omp_get_thread_num();
        size_t i1 = threadId / n_blocks;
        size_t j1 = threadId % n_blocks;

        for (size_t stage = 0; stage < n_blocks; ++stage) {
            size_t blockA = (i1 * size + ((i1 + stage) % n_blocks)) * block_size;
            size_t blockB = (((i1 + stage) % n_blocks) * size + j1) * block_size;
            size_t blockC = (i1 * size + j1) * block_size;
            
            for (size_t i = 0; i < block_size; ++i) {
                size_t strideC = blockC + i * size;  // precompute all strides
                size_t strideA = blockA + i * size;

                for (size_t j = 0; j < block_size; ++j) {
                    size_t strideB = blockB + j;
                    size_t idxC = strideC + j;

                    for (size_t k = 0; k < block_size; ++k) {
                        C->at(idxC) += A->at(strideA + k) * B->at(strideB + k * size);
                    }
                }
            }
        }
    }

    return C;
}


// Default matrix multiplication
// @param A: pointer to left matrix
// @param B: pointer to right matrix
Matrix* prod(Matrix* A, Matrix* B) {
    if (A->size() != B->size())
        return nullptr;

    size_t size = A->size();
    Matrix* C = new Matrix(size, 0);

    for (size_t i = 0; i < size; ++i) {
        size_t stride = i * size;

        for (size_t j = 0; j < size; ++j) {
            size_t idx = stride + j;

            // используем сдвиги, вместо двух индексов, т.к. в Фоксе также
            for (size_t k = 0; k < size; ++k)
                C->at(idx) += C->at(stride + k) * B->at(k * size + j);
        }
    }

    return C;
}