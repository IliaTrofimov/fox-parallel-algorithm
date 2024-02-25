#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// Create [size x size] square matrix and fill it with give value.
double* matrix_fill(size_t size, double fill);

// Create [size x size] square matrix and fill it with randome values.
double* matrix_rand(size_t size, int sup, size_t seed);

// Default square matrix multiplication. C := A*B.
void matrix_prod(double* A, double* B, double* C, size_t size);

// Check if given matrices [size x size] are equal element-wise.
int matrix_equals(double* A, double* B, size_t size);

// Print given matrix.
void matrix_print(double* A, size_t size, const char* name);
