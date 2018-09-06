#pragma once
#include <Array2D.h>

template <typename T>
class Array3D
{
public:
	// 构造
	Array3D() { }

	Array3D(int page, int col, int row, T value);

	// from vector image 64FC1
	Array3D(const vector<Mat> &vector_img);

	// *************************** 输入 ***************************************** //

	void from_vector_image_64FC1(const vector<Mat> &vector_img);

	// *************************** 赋值 ***************************************** //

	void create(int page, int col, int row, T value);

	Array2D<T>& at(int page);// 写

	const Array2D<T>& at(int page) const;// 读

	void push_back(const Array2D<T> &val);


	// *************************** 操作 ***************************************** //

	void set_zero();

	void set_value(T val);

	void set_zero_same_size_as(const Array3D<T> &array2D);

	void clear();

	// 归一化为0~1
	void normalize();

	Array3D<T> sampling(const int &sample_interval) const;

	void expand_to_full_size(int col_size, int row_size);

	vector<T> reshape_to_vector() const;

	Array2D<T> reshape_to_Array2D() const;

	// 将列向量转为n幅图像
	static vector<Array3D<T>> reshape_from_Array2D(const Array2D<T> &array2D, int channel, int col, int row)
	{
		int batch_size = array2D.size();

		vector<Array3D<T>> vec_arr3D;

		if (batch_size <= 0)
		{
			cout << "array2D is empty!" << endl << "Array3D::reshape_from_Array2D() failed!" << endl;
			return vec_arr3D;
		}

		int line_size = array2D.at(0).size();
		if (line_size != channel * col * row)
		{
			cout << "size not match!" << endl << "Array3D::reshape_from_Array2D() failed!" << endl;
			return vec_arr3D;
		}

		Array3D<T> vec_arr2D;
		Array2D<T> arr2D;

		for (int i = 0; i < channel; i++)
		{
			vec_arr2D.clear();
			for (int j = 0; j < batch_size; j++)
			{
				arr2D.clear();
				const vector<T> &v1 = array2D.at(j);
				for (int k = 0; k < col; k++)
				{
					vector<T> vec_segment(v1.begin() + i * (col * row) + k * row, v1.begin() + i * (col * row) + (k + 1) * row);
					arr2D.push_back(vec_segment);
				}
				vec_arr2D.push_back(arr2D);
			}
			vec_arr3D.push_back(vec_arr2D);
		}

		return vec_arr3D;
	}

	// ************************** 数学运算 **************************************** //

	Array3D<T> operator + (const Array3D<T> &array3D) const;

	Array3D<T> operator + (const T &val) const;

	Array3D<T> operator * (const Array3D<T> &array3D) const;

	Array3D<T> operator * (const T &val) const;

	void add(const Array3D<T> &array3D);

	void dot_product(const Array3D<T> &array3D);

	// *************************** 输出 ***************************************** //

	int size() const;

	void print() const;

	void show_specified_images_64FC1(const std::string& MultiShow_WinName, CvSize SubPlot, CvSize ImgMax_Size, int time_msec) const;

private:

	vector<Mat> to_vector_Mat_64FC1() const;

	vector<Array2D<T>> _array3D;

};

typedef Array3D<double>                Array3Dd;
