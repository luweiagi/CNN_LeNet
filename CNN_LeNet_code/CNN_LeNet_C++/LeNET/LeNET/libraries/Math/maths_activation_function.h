#pragma once
#include <vector>
#include <vector_array.h>

using namespace std;;

typedef enum {
	SoftMax = 0,
	ReLU,
} activation_function_type;


vector<array2D> activation_function(const vector<array2D> &vector_array, activation_function_type activ_func_type);

vector<array2D> soft_max(const vector<array2D> &vector_array);

vector<array2D> relu(const vector<array2D> &vector_array);
