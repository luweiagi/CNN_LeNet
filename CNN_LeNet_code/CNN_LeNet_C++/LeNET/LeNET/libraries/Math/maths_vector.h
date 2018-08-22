#pragma once

#include <vector>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

using namespace std;

typedef vector<vector<double>> array_double;


vector<double> get_vector_double_n2m(int n, int m);

vector<int> randperm_vector(int num);

array_double rand_array_double(int col, int row, double minimum, double maximum);

array_double zero_array_double(int col, int row);

array_double array_double_with_assigned_value(int col, int row, double value);

void print_array_double(const array_double &array);

array_double get_zero_array_double_same_size_as(const array_double &array);



template <typename T>
void print_vector(const vector<T> &vector_variable)
{
	for (vector<T>::iterator it = vector_variable.begin(); it != vector_variable.end(); it++)
	{
		cout << *it << endl;;
	}
}
