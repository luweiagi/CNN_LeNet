#include <Array3D.h>


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
void  Array3D<T>::from_vector_image_64FC1(const vector<Mat> &vector_img)
{
	int page = vector_img.size();

	_array3D.resize(page);

	Array2D<T> array2D;
	for (int i = 0; i < page; i++)
	{
		array2D.from_image_64FC1(vector_img.at(i));
		_array3D.at(i) = array2D;
	}
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
void Array3D<T>::normalize()
{
	for (int page = 0; page < _array3D.size(); page++)
	{
		_array3D.at(page).normalize();
	}
}


template <typename T>
int Array3D<T>::size() const
{
	return _array3D.size();
}



// only define for double
template Array3D<double>::Array3D<double>(int page, int col, int row, double value);
template void Array3D<double>::from_vector_image_64FC1(const vector<Mat> &vector_img);
template Array2D<double>& Array3D<double>::at(int page);
template const Array2D<double>& Array3D<double>::at(int page) const;
template void Array3D<double>::push_back(const Array2D<double> &val);
template void Array3D<double>::set_zero();
template void Array3D<double>::set_value(double val);
template void Array3D<double>::set_zero_same_size_as(const Array3D<double> &array3D);
template void Array3D<double>::normalize();
template int Array3D<double>::size() const;
