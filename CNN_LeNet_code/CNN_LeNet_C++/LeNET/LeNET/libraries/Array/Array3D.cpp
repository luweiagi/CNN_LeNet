#include <Array3D.h>
#include <maths_image.h>

template <typename T>
Array3D<T>::Array3D<T>(int page, int col, int row, T value)
{
	if (page <= 0 || col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "Array3D init failed!" << endl;
	}

	Array2D<T> array2D(col, row, value);
	_array3D.assign(page, array2D);
}


template <typename T>
Array3D<T>::Array3D<T>(const vector<Mat> &vector_img)
{
	int page = vector_img.size();

	if (page <= 0 || vector_img.at(0).cols <= 0 || vector_img.at(0).rows <= 0)
	{
		cout << "vector image is empty!" << endl << "Array3D constructor(const vector<Mat> &vector_img) failed!" << endl;
		return;
	}

	_array3D.resize(page);

	Array2D<T> array2D;
	for (int i = 0; i < page; i++)
	{
		array2D.from_image_64FC1(vector_img.at(i));
		_array3D.at(i) = array2D;
	}
}


template <typename T>
void Array3D<T>::from_vector_image_64FC1(const vector<Mat> &vector_img)
{
	int page = vector_img.size();

	if (page <= 0 || vector_img.at(0).cols <= 0 || vector_img.at(0).rows <= 0)
	{
		cout << "vector image is empty!" << endl << "Array3D.from_vector_image_64FC1() failed!" << endl;
		return;
	}

	_array3D.resize(page);

	Array2D<T> array2D;
	for (int i = 0; i < page; i++)
	{
		array2D.from_image_64FC1(vector_img.at(i));
		_array3D.at(i) = array2D;
	}
}


template <typename T>
void Array3D<T>::create(int page, int col, int row, T value)
{
	if (page <= 0 || col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "Array3D.create( ) failed!" << endl;
		return;
	}

	Array2D<T> array2D(col, row, value);
	_array3D.assign(page, array2D);
}


template <typename T>
Array2D<T>& Array3D<T>::at(int page)
{
	return _array3D.at(page);
}


template <typename T>
const Array2D<T>& Array3D<T>::at(int page) const
{
	return _array3D.at(page);
}


template <typename T>
void Array3D<T>::push_back(const Array2D<T> &val)
{
	if (_array3D.size() != 0)
	{
		if (_array3D.at(0).size() != val.size() || _array3D.at(0).at(0).size() != val.at(0).size())
		{
			cout << "Error: the Array2D you push back is not the same size of before!" << endl << "Array3D.push_back() failed!" << endl;
			return;
		}
	}
	_array3D.push_back(val);
}


template <typename T>
void Array3D<T>::set_zero()
{
	for (int page = 0; page < _array3D.size(); page++)
	{
		_array3D.at(page).set_zero();
	}
}


template <typename T>
void Array3D<T>::set_value(T val)
{
	for (int page = 0; page < _array3D.size(); page++)
	{
		_array3D.at(page).set_value(val);
	}
}


template <typename T>
void Array3D<T>::set_zero_same_size_as(const Array3D<T> &array3D)
{
	int page = array3D.size();

	if (page == 0)
	{
		cout << "array3D is empty!" << endl << "Array3D.set_zero_same_size_as() failde!" << endl;
		return;
	}

	Array2D<T> array2D;
	array2D.set_zero_same_size_as(array3D.at(0));

	_array3D.assign(page, array2D);
}


template <typename T>
void Array3D<T>::clear()
{
	_array3D.clear();
}


template <typename T>
void Array3D<T>::normalize()
{
	int page = _array3D.size();

	for (int i = 0; i < page; i++)
	{
		_array3D.at(i).normalize();
	}
}


template <typename T>
Array3D<T> Array3D<T>::sampling(const int &sample_interval) const
{
	int page = _array3D.size();
	if (page <= 0)
	{
		cout << "Array3D page is zero!" << endl << "Array3D.sampling() failed!" << endl;
		Array3D<T> temp;
		return temp;
	}

	Array3D<T> sampled_array3D;

	for (int i = 0; i < page; i++)
	{
		sampled_array3D.push_back(_array3D.at(i).sampling(sample_interval));
	}

	return sampled_array3D;
}


template <typename T>
void Array3D<T>::expand_to_full_size(int col_size, int row_size)
{
	int X_page = _array3D.size();
	int X_row = _array3D.at(0).at(0).size();
	int X_col = _array3D.at(0).size();

	int i, j, k;

	// 将卷积对象的尺寸扩展到full尺寸，即维度由X.size()变为X.size() + 2(Ker.size() - 1)。

	int X_expand_row = _array3D.at(0).at(0).size() + 2 * (row_size - 1);
	int X_expand_col = _array3D.at(0).size() + 2 * (col_size - 1);

	vector<Array2D<T>> temp;

	Array2D<T> X_expand(X_expand_col, X_expand_row, 0);

	for (i = 0; i < X_page; i++)
	{
		for (j = 0; j < X_row; j++)
		{
			for (k = 0; k < X_col; k++)
			{
				X_expand.at(col_size - 1 + k).at(j + row_size - 1) = _array3D.at(i).at(k).at(j);
			}
		}
		temp.push_back(X_expand);
	}

	_array3D = temp;
}


// 将三维的Array3D变成一维向量，一维向量先按页分，再按列分
template <typename T>
vector<T> Array3D<T>::reshape_to_vector() const
{
	int page = _array3D.size();
	if (page <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.reshape_to_vector() failed!" << endl;
		vector<T> temp;
		return temp;
	}

	int col = _array3D.at(0).size();
	int row = _array3D.at(0).at(0).size();

	vector<T> vec_page;
	vector<T> vec_col;
	for (int i = 0; i < page; i++)
	{
		vec_col.clear();
		for (int j = 0; j < col; j++)
		{
			vec_col.insert(vec_col.end(), _array3D.at(i).at(j).begin(), _array3D.at(i).at(j).end());
		}
		vec_page.insert(vec_page.end(), vec_col.begin(), vec_col.end());
	}
	
	return vec_page;
}


// 将3D的最后两个D变成一维向量，按照matlab的做法，是按照列的，例如
// a = [1, 2, 3;
//	    4, 5, 6];
// reshape(a, 6, 1) = [1 4 2 5 3 6]'
template <typename T>
Array2D<T> Array3D<T>::reshape_to_Array2D() const
{
	int page = _array3D.size();
	Array2D<T> reshape_2D;

	if (page == 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.reshape_to_Array2D() failed!" << endl;
		return reshape_2D;
	}

	for (int i = 0; i < page; i++)
	{
		reshape_2D.push_back(_array3D.at(i).reshape_to_vector());
	}

	return reshape_2D;
}


template <typename T>
Array3D<T> Array3D<T>::operator + (const Array3D<T> &array3D) const
{
	int page = _array3D.size();

	if (page <= 0)
	{
		return array3D;
	}

	int page_B = array3D.size();
	if (page_B < 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.operator + Array3D failed!" << endl;
		Array3D<T> temp;
		return temp;
	}

	int col = _array3D.at(0).size();
	int row = _array3D.at(0).at(0).size();

	int col_B = array3D.at(0).size();
	int row_B = array3D.at(0).at(0).size();

	if (page != page_B || col != col_B || row != row_B)
	{
		cout << "size is not same! " << endl << "Array3D.operator + Array3D failed!" << endl;
		Array3D<T> temp;
		return temp;
	}


	Array3D<T> array3D_ret;

	for (int i = 0; i < page; i++)
	{
		array3D_ret.push_back(_array3D.at(i) + array3D.at(i));
	}

	return array3D_ret;
}


template <typename T>
Array3D<T> Array3D<T>::operator + (const T &val) const
{
	int page = _array3D.size();
	if (page <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.operator + val failed!" << endl;
		Array3D<T> temp;
		return temp;
	}

	Array3D<T> array3D_ret;

	for (int i = 0; i < page; i++)
	{
		array3D_ret.push_back(_array3D.at(i) + val);
	}

	return array3D_ret;
}


template <typename T>
Array3D<T> Array3D<T>::operator * (const Array3D<T> &array3D) const
{
	int page = _array3D.size();
	int page_B = array3D.size();
	if (page <= 0 || page_B < 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.operator * Array3D failed!" << endl;
		Array3D<T> temp;
		return temp;
	}

	int col = _array3D.at(0).size();
	int row = _array3D.at(0).at(0).size();

	int col_B = array3D.at(0).size();
	int row_B = array3D.at(0).at(0).size();

	if (page != page_B || col != col_B || row != row_B)
	{
		cout << "size is not same! " << endl << "Array3D.operator * Array3D failed!" << endl;
		Array3D<T> temp;
		return temp;
	}


	Array3D<T> array3D_ret;

	for (int i = 0; i < page; i++)
	{
		array3D_ret.push_back(_array3D.at(i) * array3D.at(i));
	}

	return array3D_ret;
}


template <typename T>
Array3D<T> Array3D<T>::operator * (const T &val) const
{
	int page = _array3D.size();
	if (page <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.operator * val failed!" << endl;
		Array3D<T> temp;
		return temp;
	}

	Array3D<T> array3D_ret;

	for (int i = 0; i < page; i++)
	{
		array3D_ret.push_back(_array3D.at(i) * val);
	}

	return array3D_ret;
}


template <typename T>
void Array3D<T>::add(const Array3D<T> &array3D)
{
	int page = _array3D.size();
	int col = _array3D.at(0).size();
	int row = _array3D.at(0).at(0).size();

	int page_B = array3D.size();
	int col_B = array3D.at(0).size();
	int row_B = array3D.at(0).at(0).size();

	if (page != page_B || col != col_B || row != row_B)
	{
		cout << "size is not same! " << endl << "Array3D.add() failed!" << endl;
		return;
	}

	for (int i = 0; i < page; i++)
	{
		_array3D.at(i).add(array3D.at(i));
	}
}


template <typename T>
void Array3D<T>::dot_product(const Array3D<T> &array3D)
{
	int page = _array3D.size();
	if (page <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.dot_product() failed!" << endl;
		return;
	}

	for (int i = 0; i < page; i++)
	{
		_array3D.at(i).dot_product(array3D.at(i));
	}
}


template <typename T>
int Array3D<T>::size() const
{
	return _array3D.size();
}


template <typename T>
void Array3D<T>::print() const
{
	int page = _array3D.size();
	for (int i = 0; i < page; i++)
	{
		_array3D.at(i).print();
	}
}


template <typename T>
void Array3D<T>::show_specified_images_64FC1(const std::string& MultiShow_WinName, CvSize SubPlot, CvSize ImgMax_Size, int time_msec) const
{
	if (_array3D.size() <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.show_specified_images_64FC1() failed!" << endl;
		return;
	}

	vector_Mat_64FC1_show_one_window(MultiShow_WinName, this->to_vector_Mat_64FC1(), SubPlot, ImgMax_Size, time_msec);
}


template <typename T>
vector<Mat> Array3D<T>::to_vector_Mat_64FC1() const
{
	int size = _array3D.size();

	if (size <= 0)
	{
		cout << "Array3D is empty!" << endl << "Array3D.to_vector_Mat_64FC1() failed!" << endl;
	}

	vector<Mat> vector_Mat;

	for (int i = 0; i < size; i++)
	{
		vector_Mat.push_back(_array3D.at(i).to_Mat_64FC1());
	}

	return vector_Mat;
}


// only define for double
template Array3D<double>::Array3D<double>(int page, int col, int row, double value);
template Array3D<double>::Array3D<double>(const vector<Mat> &vector_img);
template void Array3D<double>::from_vector_image_64FC1(const vector<Mat> &vector_img);
template void Array3D<double>::create(int page, int col, int row, double value);
template Array2D<double>& Array3D<double>::at(int page);
template const Array2D<double>& Array3D<double>::at(int page) const;
template void Array3D<double>::push_back(const Array2D<double> &val);
template void Array3D<double>::set_zero();
template void Array3D<double>::set_value(double val);
template void Array3D<double>::set_zero_same_size_as(const Array3D<double> &array3D);
template void Array3D<double>::clear();
template void Array3D<double>::normalize();
template Array3D<double> Array3D<double>::sampling(const int &sample_interval) const;
template void Array3D<double>::expand_to_full_size(int col_size, int row_size);
template vector<double> Array3D<double>::reshape_to_vector() const;
template Array2D<double> Array3D<double>::reshape_to_Array2D() const;
template Array3D<double> Array3D<double>::operator + (const Array3D<double> &array3D) const;
template Array3D<double> Array3D<double>::operator + (const double &val) const;
template Array3D<double> Array3D<double>::operator * (const Array3D<double> &array3D) const;
template Array3D<double> Array3D<double>::operator * (const double &val) const;
template void Array3D<double>::add(const Array3D<double> &array3D);
template void Array3D<double>::dot_product(const Array3D<double> &array3D);
template int Array3D<double>::size() const;
template void Array3D<double>::print() const;
template void Array3D<double>::show_specified_images_64FC1(const std::string& MultiShow_WinName, CvSize SubPlot, CvSize ImgMax_Size, int time_msec) const;
template vector<Mat> Array3D<double>::to_vector_Mat_64FC1() const;


