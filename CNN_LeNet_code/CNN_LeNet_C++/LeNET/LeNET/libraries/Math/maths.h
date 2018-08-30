#pragma once

#include <math.h>
#include <maths_image.h>
#include <maths_matrix.h>
#include <maths_vector.h>
#include <maths_convolution.h>
#include <vector_array.h>
#include <maths_activation_function.h>
#include <maths_down_sample.h>

template <typename T>
void print_array(const  T *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		cout << *array++ << " ";
	}
	cout << endl;
}


void randperm_array(int serial_num[], int num);


