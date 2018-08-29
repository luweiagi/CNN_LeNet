#pragma once

#include <Array.h>
#include <maths.h>

using namespace std;

Array3Dd convolution_n_dim(const Array3Dd &X, const Array2Dd &Ker);

Array2Dd convolution_one_dim(Array2Dd X, Array2Dd Ker);

void change_X_size_to_fit_Ker(Array2Dd &X, const Array2Dd &Ker);
