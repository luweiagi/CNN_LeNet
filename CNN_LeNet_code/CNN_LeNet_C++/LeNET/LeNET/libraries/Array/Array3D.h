#pragma once
#include <Array2D.h>

template <typename T>
class Array3D
{
public:
	// 构造
	Array3D() { }

	Array3D(int page, int col, int row, T value);

	// *************************** 输入 ***************************************** //

	void from_vector_image_64FC1(const vector<Mat> &vector_img);

	// *************************** 赋值 ***************************************** //

	Array2D<T>& at(int page);// 写

	const Array2D<T>& at(int page) const;// 读

	void push_back(const Array2D<T> &val);


	// *************************** 操作 ***************************************** //

	void set_zero();

	void set_value(T val);

	void set_zero_same_size_as(const Array3D<T> &array2D);

	// 归一化为0~1
	void normalize();

	// ************************** 数学运算 **************************************** //


	// *************************** 输出 ***************************************** //

	int size() const;

private:

	vector<Array2D<T>> _array3D;

};

typedef Array3D<double>                Array3Dd;
