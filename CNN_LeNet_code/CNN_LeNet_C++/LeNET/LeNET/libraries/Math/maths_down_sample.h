#pragma once
#include <vector>
#include <vector_array.h>
#include <iostream>

using namespace std;


// 具体每个池化类型的定义可以看这里：
// https://blog.csdn.net/yangqingse/article/details/79841889
typedef enum
{
	MeanPooling = 0,
	MaxPooling,
	StochasticPooling
}down_sample_type;

vector<array2D> down_sample(const vector<array2D> &vector_array, const int sample_num, down_sample_type sample_type);

vector<array2D> down_sample_mean_pooling(const vector<array2D> &vector_array, const int sample_num);

vector<array2D> down_sample_max_pooling(const vector<array2D> &vector_array, const int sample_num);

vector<array2D> down_sample_stochastic_pooling(const vector<array2D> &vector_array, const int sample_num);

