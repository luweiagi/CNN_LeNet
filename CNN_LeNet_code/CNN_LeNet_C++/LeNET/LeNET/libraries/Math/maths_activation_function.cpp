#include <maths_activation_function.h>
#include <vector_array.h>
#include <maths.h>
#include <math.h>

Array3Dd activation_function(const Array3Dd &vector_array, activation_function_type activ_func_type)
{
	int page = vector_array.size();
	if (page == 0)
	{
		cout << "Array3Dd is empty!" << endl << "Array3Dd.activation_function() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

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


Array2Dd activation_function(const Array2Dd &vector_array, activation_function_type activ_func_type)
{
	int page = vector_array.size();
	if (page == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.activation_function() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

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
		Array2Dd temp;
		return temp;
		break;
	}
	}
}


Array3Dd soft_max(const Array3Dd &vector_array)
{
	int page = vector_array.size();
	if (page == 0)
	{
		cout << "Array3Dd is empty!" << endl << "Array3Dd.soft_max() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

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


Array2Dd soft_max(const Array2Dd &vector_array)
{
	int col = vector_array.size();
	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.soft_max() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int row = vector_array.at(0).size();

	Array2Dd vector_array_sigmoid = vector_array;

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			// sigmoid function: y = 1 / (1 + exp(-x))
			double exp_x = exp(-vector_array.at(i).at(j));
			vector_array_sigmoid.at(i).at(j) = 1 / (1 + exp_x);
		}
	}

	return vector_array_sigmoid;
}


Array3Dd relu(const Array3Dd &vector_array)
{
	int col = vector_array.size();
	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array3Dd.relu() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	// ToDo
	Array3Dd temp;
	return temp;
}


Array2Dd relu(const Array2Dd &vector_array)
{
	int col = vector_array.size();
	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.relu() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	// ToDo
	Array2Dd temp;
	return temp;
}


Array2Dd derivation(const Array2Dd &vector_array, activation_function_type activ_func_type)
{
	int col = vector_array.size();
	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.derivation() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	switch (activ_func_type)
	{
		case SoftMax:
		{
			return derivation_soft_max(vector_array);
			break;
		}
		case ReLU:
		{
			return derivation_relu(vector_array);
			break;
		}
		default:
		{
			Array2Dd temp;
			return temp;
			break;
		}
	}
}


Array2Dd derivation_soft_max(const Array2Dd &vector_array)
{
	int col = vector_array.size();

	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.derivation_soft_max() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int row = vector_array.at(0).size();


	Array2Dd vector_array_deriv = vector_array;

	double y;
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			// y.*(1-y)
			y = vector_array.at(i).at(j);
			vector_array_deriv.at(i).at(j) = y * (1 - y);
		}
	}

	return vector_array_deriv;
}


Array2Dd derivation_relu(const Array2Dd &vector_array)
{
	int col = vector_array.size();

	if (col == 0)
	{
		cout << "Array2Dd is empty!" << endl << "Array2Dd.derivation_relu() failed!" << endl;
		//Array2Dd temp;
		//return temp;
	}

	// ToDo
	Array2Dd temp;
	return temp;
}


Array3Dd derivation(const Array3Dd &vector_array, activation_function_type activ_func_type)
{
	int page = vector_array.size();
	if (page == 0)
	{
		cout << "Array3Dd is empty!" << endl << "Array3Dd.derivation() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	switch (activ_func_type)
	{
		case SoftMax:
		{
			Array3Dd vec_array = vector_array;
			vec_array.clear();
			Array2Dd array;
			for (int i = 0; i < vector_array.size(); i++)
			{
				vec_array.push_back(derivation_soft_max(vector_array.at(i)));
			}
			return vec_array;
			break;
		}
		case ReLU:
		{
			Array3Dd vec_array = vector_array;
			vec_array.clear();
			Array2Dd array;
			for (int i = 0; i < vector_array.size(); i++)
			{
				vec_array.push_back(derivation_relu(vector_array.at(i)));
			}
			return vec_array;
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