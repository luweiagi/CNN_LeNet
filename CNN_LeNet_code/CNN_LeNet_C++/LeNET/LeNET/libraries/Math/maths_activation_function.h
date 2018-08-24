#pragma once
#include <vector>
#include <vector_array.h>

using namespace std;;

typedef enum {
	SoftMax = 0,
	ReLu,
} activation_function_type;


vector<array_2D_double> activation_function(const vector<array_2D_double> &vector_array, activation_function_type activ_func_type);

vector<array_2D_double> soft_max(const vector<array_2D_double> &vector_array);

vector<array_2D_double> relu(const vector<array_2D_double> &vector_array);
