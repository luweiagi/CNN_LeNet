#pragma once

#include <vector>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector_array.h>

using namespace std;
using namespace cv;


// C++ vector的用法（整理）
// https://www.cnblogs.com/Nonono-nw/p/3462183.html


void set_target_class_one2ten(vector<vector<double>> &target_class, int length);

vector<double> get_vector_n2m(int n, int m);

vector<int> randperm_vector(int num);
// class done
array2D rand_array2D(int col, int row, double minimum, double maximum);
// class done
array2D zero_array2D(int col, int row);
// class done
array2D create_array2D(int col, int row, double value);
// class done
void print_array2D(const array2D &array);
// class done
array2D get_zero_array2D_same_size_as(const array2D &array);
// class done
array2D get_array2D_from_image_64FC1(const Mat &img);
// class2 done
vector<array2D> vector_image_64FC1_to_vector_array2D(vector<Mat> &vector_img);
// class done
void normalize_array2D_from_0_to_1(array2D &array);
// class2 done
void normalize_vector_array2D_from_0_to_1(vector<array2D> &vector_array);
// class2 done
void zero_vector_array2D(vector<array2D> &vector_array);
// class2 done
vector<array2D> get_zero_vector_array2D_same_size_as(const vector<array2D> &vector_array);
// class2 done
vector<array2D> create_vector_array2D(int vector_size, int array_col, int array_row, double value);
// class2 done
vector<array2D> get_A_add_B_vector_array2D(const vector<array2D> &vector_array_A, const vector<array2D> &vector_array_B);
// class2 done
vector<array2D> add_vector_array2D_and_num(const vector<array2D> &vector_array, const double &num);
// class done
void flip_xy_array2D(array2D &array);
// class done
array2D get_specific_size_array2D_from_specific_position(const array2D &X, int size_col, int size_row, int pos_col, int pos_row);
// class done
array2D get_A_dot_product_B_array2D(const array2D &array_A, const array2D &array_B);
// class done
double sum_of_array2D(const array2D &array);
// class2 done
vector<array2D> equal_interval_sampling_vector_array2D(const vector<array2D> &vector_array, const int sample_num);
// class2 done
vector<array2D> get_A_dot_product_B_vector_array2D(const vector<array2D> &vector_array_A, const vector<array2D> &vector_array_B);


template <typename T>
void print_vector(const vector<T> &vector_variable)
{
	for (vector<T>::const_iterator it = vector_variable.begin(); it != vector_variable.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}
