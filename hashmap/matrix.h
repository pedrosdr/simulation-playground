#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>

class matrix {
private:
    std::vector<std::vector<double>> mat;
    size_t nrows;
    size_t ncols;

public:
    matrix(size_t nrows, size_t ncols);
    matrix(double value, size_t nrows, size_t ncols);
    ~matrix();

    void print();
};

#endif