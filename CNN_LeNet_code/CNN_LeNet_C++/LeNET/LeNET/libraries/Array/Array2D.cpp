#include <Array2D.h>
#include <float.h>
#include <algorithm>
#include <time.h>
#include <maths.h>

template <typename T>
Array2D<T>::Array2D<T>(int col, int row, T value)
{
	if (col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "Array2D init failed!" << endl;
	}

	vector<T> vector_temp;
	vector_temp.assign(row, value);
	_array2D.assign(col, vector_temp);
}


template <typename T>
Array2D<T>::Array2D(const vector<vector<T>> &vec_vec)
{
	_array2D = vec_vec;
}


template <typename T>
Array2D<T>::Array2D<T>(const Mat &img)
{
	int row, col, i;
	row = img.rows;
	col = img.cols;

	if (row <= 0 || col <= 0)
	{
		cout << "image is empty!" << endl << "Array2D constructor(const Mat &img) failed!" << endl;
		return;
	}

	// 初始化_array2D的大小尺寸
	_array2D.resize(col);
	for (i = 0; i < col; i++)
	{
		_array2D.at(i).resize(row);
	}

	for (int i = 0; i < row; i++)
	{
		const double* pData = img.ptr<double>(i);	//第i+1行的所有元素
		for (int j = 0; j < col; j++)
		{
			_array2D.at(j).at(i) = (T)pData[j];
			//这行也可以：array_ret.at(j).at(i) = img.at<double>(i, j);
		}
	}
}


template <typename T>
void Array2D<T>::from_image_64FC1(const Mat &img)
{
	int row, col, i;
	row = img.rows;
	col = img.cols;

	if (row <= 0 || col <= 0)
	{
		cout << "image is empty!" << endl << "Array2D.from_image_64FC1() failed!" << endl;
		return;
	}

	// 初始化_array2D的大小尺寸
	_array2D.resize(col);
	for (i = 0; i < col; i++)
	{
		_array2D.at(i).resize(row);
	}

	for (int i = 0; i < row; i++)
	{
		const double* pData = img.ptr<double>(i);	//第i+1行的所有元素
		for (int j = 0; j < col; j++)
		{
			_array2D.at(j).at(i) = (T)pData[j];
			//这行也可以：array_ret.at(j).at(i) = img.at<double>(i, j);
		}
	}
}


template <typename T>
void Array2D<T>::create(int col, int row, T value)
{
	if (col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "Array2D.create() stoped!" << endl;
	}

	vector<T> vector_temp;
	vector_temp.assign(row, value);

	_array2D.assign(col, vector_temp);
}


template <typename T>
void Array2D<T>::zeros(int col, int row)
{
	create(col, row, 0);
}


template <typename T>
vector<T>& Array2D<T>::at(int col)
{
	return _array2D.at(col);
}


template <typename T>
const vector<T>& Array2D<T>::at(int col) const
{
	return _array2D.at(col);
}


template <typename T>
void Array2D<T>::push_back(vector<T> val)
{
	_array2D.push_back(val);
}


template <typename T>
void Array2D<T>::get_specific_patch(const Array2D<T> &array2D, int size_col, int size_row, int pos_col, int pos_row)
{
	if (size_col + pos_col > array2D.size() || size_row + pos_row > array2D.at(0).size())
	{
		cout << "pos or size has exceed range!" << endl << "Array2D.get_specific_patch() failed!" << endl;
	}

	_array2D.clear();
	_array2D.resize(size_col);

	for (int i = 0; i < size_col; i++)
	{
		for (int j = 0; j < size_row; j++)
		{
			_array2D.at(i).push_back(array2D.at(pos_col + i).at(pos_row + j));
		}
	}
}


template <typename T>
void Array2D<T>::set_zero()
{
	int col = _array2D.size();

	if (col <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.set_zero() failed!" << endl;
		return;
	}

	int row = _array2D.at(0).size();

	if (row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.set_zero() stoped!" << endl;
		return;
	}

	vector<T> vector_temp;
	vector_temp.assign(row, 0);

	_array2D.assign(col, vector_temp);
}


template <typename T>
void Array2D<T>::set_value(T val)
{
	int col = _array2D.size();

	if (col <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.set_zero() failed!" << endl;
		return;
	}

	int row = _array2D.at(0).size();

	if (row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.set_zero() stoped!" << endl;
		return;
	}

	vector<T> vector_temp;
	vector_temp.assign(row, val);

	_array2D.assign(col, vector_temp);
}


template <typename T>
void Array2D<T>::set_zero_same_size_as(const Array2D<T> &array2D)
{
	int i, j;

	_array2D.resize(array2D.size());

	for (i = 0; i < _array2D.size(); i++)
	{
		_array2D[i].assign(array2D.at(i).size(), 0);
	}
}


template <typename T>
void Array2D<T>::clear()
{
	_array2D.clear();
}


template <typename T>
void Array2D<T>::normalize()
{
	// find min and max;

	double min = DBL_MAX;
	double max = -DBL_MAX;
	double array_ij;

	for (int i = 0; i < _array2D.size(); i++)
	{
		for (int j = 0; j < _array2D.at(i).size(); j++)
		{
			array_ij = _array2D.at(i).at(j);
			if (array_ij > max)
			{
				max = array_ij;
			}
			if (array_ij < min)
			{
				min = array_ij;
			}
		}
	}

	// normalize
	for (int i = 0; i < _array2D.size(); i++)
	{
		for (int j = 0; j < _array2D.at(i).size(); j++)
		{
			if (max == min)
			{
				_array2D.at(i).at(j) = (T)1;
			}
			else
			{
				array_ij = _array2D.at(i).at(j);
				array_ij -= min;
				array_ij /= (max - min);
				_array2D.at(i).at(j) = (T)array_ij;
			}
		}
	}
}


template <typename T>
void Array2D<T>::set_rand(int col, int row, double minimum, double maximum)
{
	int i, j;

	_array2D.resize(col);

	for (i = 0; i < col; i++)
	{
		_array2D[i].resize(row);
	}

	srand((unsigned)time(NULL));//srand()函数产生一个以当前时间开始的随机种子
	for (i = 0; i < col; i++)
	{
		for (j = 0; j < row; j++)
		{
			_array2D[i][j] = minimum + (maximum - minimum) * rand() / RAND_MAX;
		}
	}
}


template <typename T>
Array2D<T> Array2D<T>::sampling(const int &sample_interval) const
{
	int col = _array2D.size();
	int row = _array2D.at(0).size();
	if (col <= 0 || row <= 0)
	{
		cout << "Array2D col/row is zero!" << endl << "Array2D.sampling() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	int new_col = int((col + (sample_interval - 1)) / sample_interval); // int(3/2) = 1
	int new_row = int((row + (sample_interval - 1)) / sample_interval);
	if (new_col <= 0 || new_row <= 0)
	{
		cout << "Array2D col/row is smaller than sample size!" << endl << "Array2D.sampling() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> sampled_array2D(new_col, new_row, 0);

	for (int i = 0; i < new_col; i++)
	{
		for (int j = 0; j < new_row; j++)
		{
			sampled_array2D.at(i).at(j) = _array2D.at(i * sample_interval).at(j * sample_interval);
		}
	}

	return sampled_array2D;
}


template <typename T>
void Array2D<T>::expand_to_full_size(int col_size, int row_size)
{
	int X_row = _array2D.at(0).size();
	int X_col = _array2D.size();

	int i, j;

	// 将卷积对象的尺寸扩展到full尺寸，即维度由X.size()变为X.size() + 2(Ker.size() - 1)。

	int X_expand_row = _array2D.at(0).size() + 2 * (row_size - 1);
	int X_expand_col = _array2D.size() + 2 * (col_size - 1);

	

	vector<T> vec_temp;
	vec_temp.assign(X_expand_row, 0);
	vector<vector<T>> temp;
	temp.assign(X_expand_col, vec_temp);

	for (i = 0; i < X_row; i++)
	{
		for (j = 0; j < X_col; j++)
		{
			temp.at(col_size - 1 + j).at(i + row_size - 1) = _array2D.at(j).at(i);
		}
	}

	_array2D = temp;
}


// 将2D变成一维向量，按照matlab的做法，是按照列的，例如
// a = [1, 2, 3;
//	    4, 5, 6];
// reshape(a, 6, 1) = [1 4 2 5 3 6]'
template <typename T>
vector<T> Array2D<T>::reshape_to_vector() const
{
	vector<T> reshape_vector;

	int col = _array2D.size();
	int row = _array2D.at(0).size();

	if (col == 0 || row == 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.reshape_to_vector() failed!" << endl;
		return reshape_vector;
	}

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			reshape_vector.push_back(_array2D.at(i).at(j));
		}
	}

	return reshape_vector;
}


template <typename T>
void Array2D<T>::append_along_row(const Array2D<T> &array2D)
{
	int col = _array2D.size();
	int new_col = array2D.size();
	if (col == 0)
	{
		_array2D.resize(new_col);
	}
	else if (col != new_col)
	{
		cout << "size not same!" << endl << "Array2D.append_along_row() failed!" << endl;
		return;
	}

	for (int i = 0; i < new_col; i++)
	{
		_array2D.at(i).insert(_array2D.at(i).end(), array2D.at(i).begin(), array2D.at(i).end());
	}
}


template <typename T>
Array2D<T> Array2D<T>::transpose() const
{
	int col = _array2D.size();
	if (col == 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.transpose() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}
	int row = _array2D.at(0).size();

	T *arr = new T[row * col]();

	int i, j;

	// 给矩阵赋值
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			// 按照行来存储矩阵
			int index = i * col + j;
			T val = _array2D.at(j).at(i);
			arr[i * col + j] = _array2D.at(j).at(i);
		}
	}

	Array2D<T> array2D_trans;
	for (i = 0; i < row; i++)
	{
		vector<T> array2D_col(arr + i * col, arr + (i + 1) * col);
		array2D_trans.push_back(array2D_col);
	}

	delete[] arr;

	return array2D_trans;
}


template <typename T>
Array2D<T> Array2D<T>::flip_xy() const
{
	vector<vector<T>>  vec_vec = _array2D;
	
	for (int i = 0; i < vec_vec.size(); i++)
	{
		reverse(vec_vec.at(i).begin(), vec_vec.at(i).end());
	}
	reverse(vec_vec.begin(), vec_vec.end());

	Array2D<T> array2D(vec_vec);
	return array2D;
}


template <typename T>
void Array2D<T>::class_0_to_9(int length)
{
	if (length < 10)
	{
		cout << "length is below 10, cann't calss 0 to 9!" << endl << " Array2D.class_0_to_9() failed!" << endl;
		return;
	}

	_array2D.clear();

	int segment_size = length / 10;
	int i, j;
	vector<double> one_hot;

	for (i = 0; i <= 8; i++)
	{
		one_hot.assign(10, 0);
		one_hot.at(i) = 1;

		for (j = 0; j < segment_size; j++)
		{
			_array2D.push_back(one_hot);
		}
	}

	// 防止length被10除不开，比如length=1003，则最后的类别10为从900到1003，而不是900到1000。
	one_hot.assign(10, 0);
	one_hot.at(i) = 1;
	for (j = i * segment_size; j < length; j++)
	{
		_array2D.push_back(one_hot);
	}
}


template <typename T>
Array2D<T> Array2D<T>::operator + (const Array2D<T> &array2D) const
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();
	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "Array2D.operator + failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> add_result(col_A, row_A, 0);

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			add_result.at(i).at(j) = _array2D.at(i).at(j) + array2D.at(i).at(j);
		}
	}

	return add_result;
}


template <typename T>
Array2D<T> Array2D<T>::operator + (const T &val) const
{
	int col = _array2D.size();
	int row = _array2D.at(0).size();

	if (col <= 0 || row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.operator + val failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> add_result(col, row, 0);

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			add_result.at(i).at(j) = _array2D.at(i).at(j) + val;
		}
	}

	return add_result;
}


template <typename T>
Array2D<T> Array2D<T>::operator - (const Array2D<T> &array2D) const
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();
	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "Array2D.operator - failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> sub_result(col_A, row_A, 0);

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			sub_result.at(i).at(j) = _array2D.at(i).at(j) - array2D.at(i).at(j);
		}
	}

	return sub_result;
}


template <typename T>
Array2D<T> Array2D<T>::operator * (const Array2D<T> &array2D) const
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();
	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "Array2D.operator * failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> dot_product_result(col_A, row_A, 0);
	dot_product_result.set_zero();

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			dot_product_result.at(i).at(j) = _array2D.at(i).at(j) * array2D.at(i).at(j);
		}
	}

	return dot_product_result;
}


template <typename T>
Array2D<T> Array2D<T>::operator * (const T &val) const
{
	int col = _array2D.size();
	int row = _array2D.at(0).size();

	if (col <= 0 || row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.operator * val failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> product_result(col, row, 0);

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			product_result.at(i).at(j) = _array2D.at(i).at(j) * val;
		}
	}

	return product_result;
}


template <typename T>
void Array2D<T>::add(const Array2D<T> &array2D)
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();
	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "Array2D.add() failed!" << endl;
		return;
	}

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			_array2D.at(i).at(j) += array2D.at(i).at(j);
		}
	}
}


template <typename T>
void Array2D<T>::dot_product(const Array2D<T> &array2D)
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();
	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "Array2D.dot_product() failed!" << endl;
		return;
	}

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			_array2D.at(i).at(j) = _array2D.at(i).at(j) * array2D.at(i).at(j);
		}
	}
}


template <typename T>
Array2D<T> Array2D<T>::product(const Array2D<T> &array2D) const
{
	int col_A = _array2D.size();
	int row_A = _array2D.at(0).size();

	int col_B = array2D.size();
	int row_B = array2D.at(0).size();

	if (col_A != row_B)
	{
		cout << "size not match!" << endl << "Array2D.product() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	if (col_A == 0 || row_A == 0 || col_B == 0 || row_B == 0)
	{
		cout << "Array2D A or B is empty!" << endl << "Array2D.product() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	T *arr_A = new T[row_A * col_A]();
	T *arr_B = new T[row_B * col_B]();

	Array2Dd AB(col_B, row_A, 0);

	int col_max = col_A > col_B ? col_A : col_B;
	int row_max = row_A > row_B ? row_A : row_B;

	int i, j, k;

	// 给A和B矩阵赋值
	for (i = 0; i < row_max; i++)
	{
		for (j = 0; j < col_max; j++)
		{
			// 给A矩阵赋值
			if ((i < row_A) && (j < col_A))
			{
				arr_A[i * col_A + j] = _array2D.at(j).at(i);
			}

			// 给B矩阵赋值
			if ((i < row_B) && (j < col_B))
			{
				arr_B[i * col_B + j] = array2D.at(j).at(i);
			}
		}
	}

	// AB = A * B
	T val_ij = 0;
	for (i = 0; i < row_A; i++)
	{
		for (j = 0; j < col_B; j++)
		{
			// arr_A[i * col_A + j]
			for (k = 0; k < col_A; k++)
			{
				val_ij += arr_A[i * col_A + k] * arr_B[k * col_B + j];
			}
			AB.at(j).at(i) = val_ij;
			val_ij = 0;
		}
	}

	delete[] arr_A;
	delete[] arr_B;

	return AB;
}


template <typename T>
T Array2D<T>::sum() const
{
	int col = _array2D.size();
	int row = _array2D.at(0).size();

	T sum_ret = 0;

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			sum_ret += _array2D.at(i).at(j);
		}
	}

	return sum_ret;
}


template <typename T>
vector<T> Array2D<T>::mean() const
{
	// 按列求平均，结果为列向量
	int col = _array2D.size();
	if (col <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.mean() failed!" << endl;
		vector<T> temp;
		return temp;
	}

	int row = _array2D.at(0).size();
	if (row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.mean() failed!" << endl;
		vector<T> temp;
		return temp;
	}

	vector<T> vec_ret;
	vec_ret.assign(row, 0);

	int i;
	for (i = 0; i < col; ++i)
	{
		vec_ret = vec_ret + (vector<T>)_array2D.at(i);
	}

	for (i = 0; i < row; ++i)
	{
		vec_ret.at(i) = vec_ret.at(i) / col;
	}

	return vec_ret;
}


template <typename T>
Array2D<T> Array2D<T>::pow(const int power) const
{
	int col = _array2D.size();
	if (col <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.pow() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	int row = _array2D.at(0).size();
	if (row <= 0)
	{
		cout << "Array2D is empty!" << endl << "Array2D.pow() failed!" << endl;
		Array2D<T> temp;
		return temp;
	}

	Array2D<T> pow_ret(col, row, 0);

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			pow_ret.at(i).at(j) = std::pow(_array2D.at(i).at(j), power);
		}
	}

	return pow_ret;
}


template <typename T>
int Array2D<T>::size() const
{
	return _array2D.size();
}


template <typename T>
void Array2D<T>::print() const
{
	int i, j;

	cout << endl;

	for (i = 0; i < _array2D.at(0).size(); i++)
	{
		for (j = 0; j < _array2D.size(); j++)
		{
			cout << _array2D.at(j).at(i) << " ";
		}
		cout << endl;
	}
}


template <typename T>
void Array2D<T>::show_image_64FC1() const
{
	Mat image = to_Mat_64FC1();

	// 显示图片   
	imshow("img", image);
	// 等待time_msec后窗口自动关闭    
	waitKey(3000);

	destroyWindow("img");
}


template <typename T>
void Array2D<T>::show_image_64FC1(int time_msec) const
{
	Mat image = to_Mat_64FC1();

	// 显示图片   
	imshow("img", image);
	// 等待time_msec后窗口自动关闭    
	waitKey(time_msec);

	destroyWindow("img");
}


template <typename T>
Mat Array2D<T>::to_Mat_64FC1() const
{
	if (_array2D.empty())
	{
		cout << "Array2D is empty!" << endl << "Array2D.to_Mat_64FC1() failed" << endl;
		Mat img;
		return img;
	}

	int row = _array2D.at(0).size();
	int col = _array2D.size();

	Mat img(row, col, CV_64FC1);
	double *ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = _array2D.at(j).at(i);
		}
	}
	return img;
}







// only define for double
template Array2D<double>::Array2D<double>(int col, int row, double value);
template Array2D<double>::Array2D<double>(const vector<vector<double>> &vec_vec);
template Array2D<double>::Array2D<double>(const Mat &img);
template void Array2D<double>::from_image_64FC1(const Mat &img);
template void Array2D<double>::create(int col, int row, double value);
template void Array2D<double>::zeros(int col, int row);
template vector<double>& Array2D<double>::at(int col);
template const vector<double>& Array2D<double>::at(int col) const;
template void Array2D<double>::push_back(vector<double> val);
template void Array2D<double>::get_specific_patch(const Array2D<double> &array2D, int size_col, int size_row, int pos_col, int pos_row);
template void Array2D<double>::set_zero();
template void Array2D<double>::set_value(double val);
template void Array2D<double>::set_zero_same_size_as(const Array2D<double> &array2D);
template void Array2D<double>::clear();
template void Array2D<double>::normalize();
template void Array2D<double>::set_rand(int col, int row, double minimum, double maximum);
template Array2D<double> Array2D<double>::sampling(const int &sample_interval) const;
template void Array2D<double>::expand_to_full_size(int col_size, int row_size);
template vector<double> Array2D<double>::reshape_to_vector() const;
template void Array2D<double>::append_along_row(const Array2D<double> &array2D);
template Array2D<double> Array2D<double>::transpose() const;
template Array2D<double> Array2D<double>::flip_xy() const;
template void Array2D<double>::class_0_to_9(int length);
template Array2D<double> Array2D<double>::operator + (const Array2D<double> &array2D) const;
template Array2D<double> Array2D<double>::operator + (const double &val) const;
template Array2D<double> Array2D<double>::operator - (const Array2D<double> &array2D) const;
template Array2D<double> Array2D<double>::operator * (const Array2D<double> &array2D) const;
template Array2D<double> Array2D<double>::operator * (const double &val) const;
template void Array2D<double>::add(const Array2D<double> &array2D);
template void Array2D<double>::dot_product(const Array2D<double> &array2D);
template Array2D<double> Array2D<double>::product(const Array2D<double> &array2D) const;
template double Array2D<double>::sum() const;
template vector<double> Array2D<double>::mean() const;
template Array2D<double> Array2D<double>::pow(const int power) const;
template int Array2D<double>::size() const;
template void Array2D<double>::print() const;
template void Array2D<double>::show_image_64FC1() const;
template void Array2D<double>::show_image_64FC1(int time_msec) const;
template Mat Array2D<double>::to_Mat_64FC1() const;



