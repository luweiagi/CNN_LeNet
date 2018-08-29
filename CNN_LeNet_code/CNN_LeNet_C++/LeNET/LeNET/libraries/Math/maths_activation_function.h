#pragma once
#include <vector>
#include <Array.h>


using namespace std;;

typedef enum {
	SoftMax = 0,
	ReLU,
} activation_function_type;


Array3Dd activation_function(const Array3Dd &vector_array, activation_function_type activ_func_type);

Array3Dd soft_max(const Array3Dd &vector_array);

Array3Dd relu(const Array3Dd &vector_array);
