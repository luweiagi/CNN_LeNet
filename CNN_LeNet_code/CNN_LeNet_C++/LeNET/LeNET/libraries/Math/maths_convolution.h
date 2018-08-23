#pragma once

#include <vector_array.h>
#include <maths.h>

using namespace std;

vector<array_2D_double> convolution_n_dim(const vector<array_2D_double> &X, const array_2D_double &Ker);

array_2D_double convolution_one_dim(array_2D_double X, array_2D_double Ker);

void change_X_size_to_fit_Ker(array_2D_double &X, const array_2D_double &Ker);
