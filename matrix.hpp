#pragma once

#include <iostream>
#include <ostream>

// Square matrix. OOP wrapper.
class Matrix {
private:
    double* _data;
    size_t _size;
    size_t _sizeSqr;

public:
    Matrix(double* data, size_t size) {
        this->_size = size;
        this->_sizeSqr = size * size;
        _data = data;
    }

    Matrix(size_t size, double fill = 0) {
        this->_size = size;
        this->_sizeSqr = size * size;

        _data = new double[_sizeSqr];
        for (size_t i = 0; i < _sizeSqr; ++i)
            _data[i] = fill;
    }

    ~Matrix() {
        delete[] _data;
    }


    // Get raw data array.
    inline double* data() const {
        return _data;
    }

    // Element at given offset.
    inline double& at(size_t idx) {
        if (idx >= _sizeSqr) throw new std::exception("Matrix index is out of range.");
        return _data[idx];
    }

    // Element at row i and column j.
    inline double& at(size_t i, size_t j) {
        return at(i * _size + j);
    }

    // Matrix size.
    inline size_t size() const {
        return _size;
    }

    // Matrix size squared.
    inline size_t size_sqr() const {
        return _sizeSqr;
    }

    // Print matrix.
    void print(std::ostream& os, const char* name = nullptr) const {
        if (name != nullptr) {
            os << name << " [" << _size << " x " << _size << "]\n";
        }
        for (size_t i = 0; i < _size; ++i) {
            size_t stride = i * _size;
            for (size_t j = 0; j < _size; ++j)
                os << _data[stride + j] << "\t";
            os << "\n";
        }
    }

    // Check if matricies are equal.
    bool equals(Matrix* other) const {
        if (other->_size != this->_size)
            return false;
        
        for (size_t i = 0; i < _sizeSqr; ++i)
            if (this->_data[i] != other->_data[i]) return false;

        return true;
    }
};


// Default matrix multiplication
// @param A: pointer to left matrix
// @param B: pointer to right matrix
Matrix* prod(Matrix* A, Matrix* B) {
    if (A->size() != B->size())
        return nullptr;

    size_t size = A->size();
    Matrix* C = new Matrix(size, 0);

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k)
                C->at(i, j) += A->at(i, k) * B->at(k, j);
        }
    }

    return C;
}

// [Optimized] Default matrix multiplication. Uses linear matrix representation.
// @param A: pointer to left matrix
// @param B: pointer to right matrix
Matrix* prod_opt(Matrix* A, Matrix* B) {
    if (A->size() != B->size())
        return nullptr;

    size_t size = A->size();
    Matrix* C = new Matrix(size, 0);

    for (size_t i = 0; i < size; ++i) {
        size_t stride = i * size;

        for (size_t j = 0; j < size; ++j) {
            size_t idx = stride + j;
            for (size_t k = 0; k < size; ++k)
                C->at(idx) += C->at(stride + k) * B->at(k * size + j);
        }
    }

    return C;
}

// [Optimized] Default matrix multiplication. Uses linear matrix representation.
// @param A: pointer to left matrix
// @param B: pointer to right matrix
// @param C: pointer to result matrix
void prod_opt(Matrix* A, Matrix* B, Matrix*& C) {
    if (A->size() != B->size())
        return;

    size_t size = A->size();
    C = new Matrix(size, 0);

    for (size_t i = 0; i < size; ++i) {
        size_t stride = i * size;

        for (size_t j = 0; j < size; ++j) {
            size_t idx = stride + j;
            for (size_t k = 0; k < size; ++k)
                C->at(idx) += C->at(stride + k) * B->at(k * size + j);
        }
    }
}