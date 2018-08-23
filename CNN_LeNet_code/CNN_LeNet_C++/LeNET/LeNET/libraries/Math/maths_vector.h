#pragma once

#include <vector>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector_array.h>

using namespace std;

using namespace cv;


void set_target_class_one2ten(vector<vector<double>> &target_class, int length);

vector<double> get_vector_double_n2m(int n, int m);

vector<int> randperm_vector(int num);

array_2D_double rand_array_2D_double(int col, int row, double minimum, double maximum);

array_2D_double zero_array_2D_double(int col, int row);

array_2D_double create_array_2D_double(int col, int row, double value);

void print_array_2D_double(const array_2D_double &array);

array_2D_double get_zero_array_2D_double_same_size_as(const array_2D_double &array);

array_2D_double image_64FC1_to_array_2D_double(const Mat &img);

vector<array_2D_double> vector_image_64FC1_to_vector_array_2D_double(vector<Mat> &vector_img);

void normalize_array_2D_double_from_0_to_1(array_2D_double &array);

void normalize_vector_array_2D_double_from_0_to_1(vector<array_2D_double> &vector_array);

void zero_vector_array_2D_double(vector<array_2D_double> &vector_array);

vector<array_2D_double> get_zero_vector_array_2D_double_same_size_as(const vector<array_2D_double> &vector_array);

vector<array_2D_double> create_vector_array_2D_double(int vector_size, int array_row, int array_col, double value);

void add_B_to_A_vector_array_2D_double(vector<array_2D_double> &vector_array_A, const vector<array_2D_double> &vector_array_B);

void flip_xy_array_2D_double(array_2D_double &array);

array_2D_double get_specific_size_array_2D_double_from_specific_position(const array_2D_double &X, int size_col, int size_row, int pos_col, int pos_row);

array_2D_double get_A_dot_product_B_array_2D_double(const array_2D_double &array_A, const array_2D_double &array_B);

double sum_of_array_2D_double(const array_2D_double &array);


template <typename T>
void print_vector(const vector<T> &vector_variable)
{
	for (vector<T>::iterator it = vector_variable.begin(); it != vector_variable.end(); it++)
	{
		cout << *it << endl;;
	}
}
