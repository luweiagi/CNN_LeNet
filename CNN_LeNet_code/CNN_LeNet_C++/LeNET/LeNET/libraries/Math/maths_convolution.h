#pragma once

#include <vector_array.h>
#include <maths.h>

using namespace std;

vector<array2D> convolution_n_dim(const vector<array2D> &X, const array2D &Ker);

array2D convolution_one_dim(array2D X, array2D Ker);

void change_X_size_to_fit_Ker(array2D &X, const array2D &Ker);
