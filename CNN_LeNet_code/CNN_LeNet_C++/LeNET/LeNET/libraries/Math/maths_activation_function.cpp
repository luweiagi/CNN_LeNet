#include <maths_activation_function.h>
#include <vector_array.h>
#include <maths.h>
#include <math.h>


vector<array_2D_double> activation_function(const vector<array_2D_double> &vector_array, activation_function_type activ_func_type)
{
	switch (activ_func_type)
	{
		case SoftMax:
		{
			return soft_max(vector_array);
			break;
		}
		case ReLu:
		{
			return relu(vector_array);
			break;
		}
		default:
		{
			// ∑µªÿ¡„
			return get_zero_vector_array_2D_double_same_size_as(vector_array);
			break;
		}
	}
}


vector<array_2D_double> soft_max(const vector<array_2D_double> &vector_array)
{
	int vector_size = vector_array.size();
	int array_col = vector_array.at(0).size();
	int array_row = vector_array.at(0).at(0).size();

	vector<array_2D_double> vector_array_sigmoid = vector_array;

	for (int i = 0; i < vector_size; i++)
	{
		for (int j = 0; j < array_col; j++)
		{
			for (int k = 0; k < array_row; k++)
			{
				// sigmoid function: y = 1 / (1 + exp(-x))
				double exp_x = exp(- vector_array.at(i).at(j).at(k));
				vector_array_sigmoid.at(i).at(j).at(k) = 1 / (1 + exp_x);
			}
		}
	}

	return vector_array_sigmoid;
}


vector<array_2D_double> relu(const vector<array_2D_double> &vector_array)
{
	// ToDo
	// ∑µªÿ¡„
	return get_zero_vector_array_2D_double_same_size_as(vector_array);
}
