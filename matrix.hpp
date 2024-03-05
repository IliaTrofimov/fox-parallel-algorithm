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
