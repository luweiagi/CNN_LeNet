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


vector<double> get_vector_n2m(int n, int m);

vector<int> randperm_vector(int num);

template<typename T>
vector<T> operator +(const vector<T> &vec_1, const vector<T> &vec_2)
{
	if (vec_1.size() == 0 || vec_2.size() == 0 || (vec_1.size() != vec_2.size()))
	{
		cout << "size is zero or not same!" << endl << "operator +(vector<T>, vector<T>) failed!" << endl;
		vector<T> temp;
		return temp;
	}

	vector<T> sum = vec_1;
	int size = sum.size();
	for (int i = 0; i < size; ++i)
	{
		sum.at(i) = vec_1.at(i) + vec_2.at(i);
	}

	return sum;
}


template<typename T>
vector<T> operator -(const vector<T> &vec_1, const vector<T> &vec_2)
{
	if (vec_1.size() == 0 || vec_2.size() == 0 || (vec_1.size() != vec_2.size()))
	{
		cout << "size is zero or not same!" << endl << "operator +(vector<T>, vector<T>) failed!" << endl;
		vector<T> temp;
		return temp;
	}

	vector<T> sub = vec_1;
	int size = sub.size();
	for (int i = 0; i < size; ++i)
	{
		sub.at(i) = vec_1.at(i) - vec_2.at(i);
	}

	return sub;
}


template<typename T>
vector<T> operator *(const vector<T> &vec_1, const vector<T> &vec_2)
{
	if (vec_1.size() == 0 || vec_2.size() == 0 || (vec_1.size() != vec_2.size()))
	{
		cout << "size is zero or not same!" << endl << "operator *(vector<T>, vector<T>) failed!" << endl;
		vector<T> temp;
		return temp;
	}

	vector<T> dot_product = vec_1;
	int size = dot_product.size();
	for (int i = 0; i < size; ++i)
	{
		dot_product.at(i) = vec_1.at(i) * vec_2.at(i);
	}

	return dot_product;
}


template<typename T>
vector<T> operator *(const vector<T> &vec, const T val)
{
	if (vec.size() == 0)
	{
		cout << "size is zero!" << endl << "operator *(vector<T>, T) failed!" << endl;
		vector<T> temp;
		return temp;
	}

	vector<T> dot_product = vec;
	int size = dot_product.size();
	for (int i = 0; i < size; ++i)
	{
		dot_product.at(i) = vec.at(i) * val;
	}

	return dot_product;
}


template<typename T>
T sum_vector(const vector<T> vec)
{
	if (vec.size() <= 0)
	{
		cout << "size is zero!" << endl << "sum_vector() failed!" << endl;
		return 0;
	}

	T sum = 0;
	int size = vec.size();
	for (int i = 0; i < size; ++i)
	{
		sum += vec.at(i);
	}

	return sum;
}


template <typename T>
void print(const vector<T> &vec)
{
	if (vec.size() == 0)
	{
		cout << "vector is empty!" << endl << "vector.print() failed!" << endl;
		return;
	}

	for (vector<T>::const_iterator it = vec.begin(); it != vec.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}


// 返回由非零值的坐标构成的向量
template <typename T>
vector<int> find(const vector<T> &vec)
{
	int size = vec.size();

	if (size <= 0)
	{
		cout << "size is zero!" << endl << "find() failed!" << endl;
		vector<int> temp;
		return temp;
	}

	vector<int> vec_no_zero_index;

	for (int i = 0; i < size; i++)
	{
		if (vec.at(i) != 0)
		{
			vec_no_zero_index.push_back(i);
		}
	}

	return vec_no_zero_index;
}