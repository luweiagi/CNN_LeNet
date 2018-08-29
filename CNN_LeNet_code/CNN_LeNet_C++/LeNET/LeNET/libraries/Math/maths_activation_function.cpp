#include <maths_activation_function.h>
#include <vector_array.h>
#include <maths.h>
#include <math.h>

Array3Dd activation_function(const Array3Dd &vector_array, activation_function_type activ_func_type)
{
	switch (activ_func_type)
	{
		case SoftMax:
		{
			return soft_max(vector_array);
			break;
		}
		case ReLU:
		{
			return relu(vector_array);
			break;
		}
		default:
		{
			Array3Dd temp;
			return temp;
			break;
		}
	}
}


Array3Dd soft_max(const Array3Dd &vector_array)
{
	int page = vector_array.size();
	int col = vector_array.at(0).size();
	int row = vector_array.at(0).at(0).size();

	Array3Dd vector_array_sigmoid = vector_array;

	for (int i = 0; i < page; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < row; k++)
			{
				// sigmoid function: y = 1 / (1 + exp(-x))
				double exp_x = exp(- vector_array.at(i).at(j).at(k));
				vector_array_sigmoid.at(i).at(j).at(k) = 1 / (1 + exp_x);
			}
		}
	}

	return vector_array_sigmoid;
}


Array3Dd relu(const Array3Dd &vector_array)
{
	// ToDo
	Array3Dd temp;
	return temp;
}
