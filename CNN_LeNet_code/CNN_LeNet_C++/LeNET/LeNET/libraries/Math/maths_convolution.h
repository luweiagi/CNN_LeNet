#pragma once

#include <Array.h>
#include <maths.h>
#include <string>

using namespace std;

Array3Dd convolution(Array3Dd X, const Array2Dd &Ker, string shape);

Array2Dd convolution(const Array3Dd &X, const Array3Dd &Ker, string shape);

Array2Dd convolution(Array2Dd X, Array2Dd Ker, string shape);
