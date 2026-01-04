#include <iostream>
#include <vector>
#include "matrix.h"

matrix::matrix(size_t nrows, size_t ncols):
nrows(nrows), ncols(ncols) 
{
    mat = std::vector<std::vector<double>>(
        nrows, std::vector<double>(ncols)
    );
}

matrix::matrix(double value, size_t nrows, size_t ncols):
nrows(nrows), ncols(ncols) 
{
    mat = std::vector<std::vector<double>>(
        nrows, std::vector<double>(ncols, value)
    );
}

matrix::~matrix() {}

void matrix::print() {
    std::cout << "[";
    for(size_t i = 0; i < nrows; i++) {
        if(i > 0) {
            std::cout << " ";
        }
        std::cout << "[";
        for(size_t j = 0; j < ncols; j++) {
            std::cout << mat[i][j];
            if(j != ncols-1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if(i != ncols - 1) {
            std::cout << "\n";
        }
    }
    std::cout << "]";
}