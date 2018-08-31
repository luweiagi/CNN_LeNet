#pragma once
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  


using namespace std;
using namespace cv;

template <typename T>
class Array2D
{
public:

	// 构造
	Array2D() { }

	Array2D(int col, int row, T value);

	// from image 64FC1
	Array2D(const Mat &img);

	// *************************** 输入 ***************************************** //

	void from_image_64FC1(const Mat &img);

	// *************************** 赋值 ***************************************** //

	void create(int col, int row, T value);

	void zeros(int col, int row);

	vector<T>& at(int col);// 写

	const vector<T>& at(int col) const;// 读

	void push_back(vector<T> val);

	void get_specific_patch(const Array2D<T> &array2D, int size_col, int size_row, int pos_col, int pos_row);

	// *************************** 操作 ***************************************** //

	void set_zero();

	void set_value(T val);

	void set_zero_same_size_as(const Array2D<T> &array2D);

	void clear();

	// 归一化为0~1
	void normalize();

	// 重置为指定范围和大小的随机数
	void set_rand(int col, int row, double minimum, double maximum);

	Array2D<T> sampling(const int &sample_interval) const;

	vector<T> reshape_to_vector() const;

	void append_along_row(const Array2D<T> &array2D);

	// 这里将一维的vector看作是列向量，和matlab函数repmat一致
	static Array2D<T> repmat(const vector<T> &vec, int row, int col)// 静态成员函数
	{
		if (row <= 0 || col <= 0 || vec.size() <= 0)
		{
			cout << "size is zero!" << endl << "Array2D.repmat() failed!" << endl;
			Array2D<T> temp;
			return temp;
		}

		vector<T> vec_row;
		Array2D<T> array2D;

		int i;
		for (i = 0; i < row; i++)
		{
			vec_row.insert(vec_row.end(), vec.begin(), vec.end());
		}

		for (i = 0; i < col; i++)
		{
			array2D.push_back(vec_row);
		}

		return array2D;
	}

	// 将Array2D的row转成Array3D的col和row，Array2D的col转为Array3D的page
	//Array3D<T> reshape(int col, int row) const;

	// 矩阵转置
	Array2D<T> transpose() const;

	// 用于卷积核的翻转
	void flip_xy();

	void class_0_to_9(int length);

	// ************************** 数学运算 **************************************** //

	Array2D<T> operator + (const Array2D<T> &array2D) const;

	Array2D<T> operator + (const T &val) const;

	Array2D<T> operator - (const Array2D<T> &array2D) const;

	// 点乘
	Array2D<T> operator * (const Array2D<T> &array2D) const;

	Array2D<T> operator * (const T &val) const;

	void add(const Array2D<T> &array2D);

	void dot_product(const Array2D<T> &array2D);

	// 矩阵乘法
	Array2D<T> product(const Array2D<T> &array2D) const;

	T sum() const;

	Array2D<T> pow(const int power_num) const;

	// *************************** 输出 ***************************************** //

	int size() const;

	void print() const;

	void show_image_64FC1() const;

	void show_image_64FC1(int time_msec) const;

	Mat to_Mat_64FC1() const;

private:

	vector<vector<T>> _array2D;

};

typedef Array2D<double>                Array2Dd;
