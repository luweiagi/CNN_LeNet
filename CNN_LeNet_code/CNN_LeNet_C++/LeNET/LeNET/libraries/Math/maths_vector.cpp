#include <maths_vector.h>
#include <iostream>
#include <time.h>
#include<algorithm>


void set_target_class_one2ten(vector<vector<double>> &target_class, int length)
{
	int segment_size = length / 10;
	int i, j;
	vector<double> one_hot;

	for (i = 0; i <= 8; i++)
	{
		one_hot.assign(10, 0);
		one_hot.at(i) = 1;

		for (j = 0; j < segment_size; j++)
		{
			target_class.push_back(one_hot);
		}
	}

	// 防止length被10除不开，比如length=1003，则最后的类别10为从900到1003，而不是900到1000。
	one_hot.assign(10, 0);
	one_hot.at(i) = 1;
	for (j = i * segment_size; j < length; j++)
	{
		target_class.push_back(one_hot);
	}
}


// 得到n~m的vector
vector<double> get_vector_n2m(int n, int m)
{
	vector<double> vector_n2m;

	if (n > m)
	{
		cout << "n > m in function get_vector_double_n2m, so do nothing and return!" << endl;
	}

	for (int i = n; i <= m; i++)
	{
		vector_n2m.push_back(i);
	}

	return vector_n2m;
}


// 类似matlab中的randperm函数，即得到0~num-1之间的打乱顺序后的数组
vector<int> randperm_vector(int num)
{
	vector<int> serial_num;

	for (int i = 0; i < num; i++)
	{
		serial_num.push_back(i);
	}

	int j, temp;

	srand((unsigned)time(NULL));//srand()函数产生一个以当前时间开始的随机种子
	for (int i = num; i > 1; i--)
	{
		j = rand() % i;
		temp = serial_num.at(i - 1);
		serial_num.at(i - 1) = serial_num.at(j);
		serial_num.at(j) = temp;
	}

	return serial_num;
}



// 注意：col 是该矩阵的列， row是该矩阵的行
array2D rand_array2D(int col, int row, double minimum, double maximum)
{
	int i, j;

	array2D array(col);

	for (i = 0; i < col; i++)
	{
		array[i].resize(row);
	}

	for (i = 0; i < array.size(); i++)
	{
		for (j = 0; j < array[0].size(); j++)
		{
			array[i][j] = minimum + (maximum - minimum) * rand() / RAND_MAX;
		}
	}

	return array;
}


// 注意：col 是该矩阵的列， row是该矩阵的行
array2D zero_array2D(int col, int row)
{
	return create_array2D(col, row, 0);
}


// 注意：col 是该矩阵的列， row是该矩阵的行
array2D create_array2D(int col, int row, double value)
{
	if (col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "create_array2D( ) stoped!" << endl;
	}

	vector<double> vector_temp;
	vector_temp.assign(row, value);

	array2D array2D_ret;
	array2D_ret.assign(col, vector_temp);

	return array2D_ret;
}


void print_array2D(const array2D &array)
{
	int i, j;

	for (i = 0; i < array[0].size(); i++)
	{
		for (j = 0; j < array.size(); j++)
		{
			cout << array[j][i] << " ";
		}

		cout << endl;
	}
}


array2D get_zero_array2D_same_size_as(const array2D &array)
{
	int i, j;

	array2D array_ret = array;

	for (i = 0; i < array_ret.size(); i++)
	{
		array_ret[i].assign(array[i].size(), 0);
	}

	return array_ret;
}


// 把图片转为array2D
array2D get_array2D_from_image_64FC1(const Mat &img)
{
	int row, col, i;
	row = img.rows;
	col = img.cols;

	// 初始化array_ret的大小尺寸
	array2D array_ret(col);
	for (i = 0; i < col; i++)
	{
		array_ret.at(i).resize(row);
	}

	for (int i = 0; i < row; i++)
	{
		const double* pData = img.ptr<double>(i);	//第i+1行的所有元素
		for (int j = 0; j < col; j++)
		{
			array_ret.at(j).at(i) = pData[j];
			//这行也可以：array_ret.at(j).at(i) = img.at<double>(i, j);
		}
	}

	return array_ret;
}


vector<array2D> vector_image_64FC1_to_vector_array2D(vector<Mat> &vector_img)
{
	int size = vector_img.size();

	vector<array2D> vector_array(size);

	for (int i = 0; i < size; i++)
	{
		vector_array.at(i) = get_array2D_from_image_64FC1(vector_img.at(i));
	}

	return vector_array;
}


void normalize_array2D_from_0_to_1(array2D &array)
{
	// find min and max;
	double min =  1000000;
	double max = -1000000;

	for (int i = 0; i < array.size(); i++)
	{
		for (int j = 0; j < array.at(i).size(); j++)
		{
			double array_ij = array.at(i).at(j);
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
	for (int i = 0; i < array.size(); i++)
	{
		for (int j = 0; j < array.at(i).size(); j++)
		{
			array.at(i).at(j) -= min;
			array.at(i).at(j) /= (max - min);
		}
	}
}


void normalize_vector_array2D_from_0_to_1(vector<array2D> &vector_array)
{
	for (int i = 0; i < vector_array.size(); i++)
	{
		normalize_array2D_from_0_to_1(vector_array.at(i));
	}
}


void zero_vector_array2D(vector<array2D> &vector_array)
{
	array2D array2D_zero = get_zero_array2D_same_size_as(vector_array.at(1));
	vector_array.assign(vector_array.size(), array2D_zero);
}


vector<array2D> get_zero_vector_array2D_same_size_as(const vector<array2D> &vector_array)
{
	vector<array2D>array_ret = vector_array;

	zero_vector_array2D(array_ret);

	return array_ret;
}


vector<array2D> create_vector_array2D(int vector_size, int array_col, int array_row, double value)
{
	if (vector_size <= 0 || array_col <= 0 || array_row <= 0)
	{
		cout << "size < 0 !" << endl << "create_vector_array2D( ) stoped!" << endl;
	}

	vector<double> vector_temp;
	vector_temp.assign(array_row, value);

	array2D array2D_temp;
	array2D_temp.assign(array_col, vector_temp);

	vector<array2D> array_3D_ret;
	array_3D_ret.assign(vector_size, array2D_temp);

	return array_3D_ret;
}


vector<array2D> get_A_add_B_vector_array2D(const vector<array2D> &vector_array_A, const vector<array2D> &vector_array_B)
{
	int vector_size = vector_array_A.size();
	int array_col = vector_array_A.at(0).size();
	int array_row = vector_array_A.at(0).at(0).size();

	int vector_size_B = vector_array_B.size();
	int array_col_B = vector_array_B.at(0).size();
	int array_row_B = vector_array_B.at(0).at(0).size();

	if (vector_size != vector_size_B || array_col != array_col_B || array_row != array_row_B)
	{
		cout << "vector_array_A size is not same to vector_array_B size! " << endl << "add_vector_array2D() stoped!" << endl;
		return vector_array_A;
	}

	vector<array2D> vector_array_sum = vector_array_A;

	for (int i = 0; i < vector_size; i++)
	{
		for (int j = 0; j < array_col; j++)
		{
			for (int k = 0; k < array_row; k++)
			{
				vector_array_sum.at(i).at(j).at(k) = vector_array_A.at(i).at(j).at(k) + vector_array_B.at(i).at(j).at(k);
			}
		}
	}

	return vector_array_sum;
}


vector<array2D> add_vector_array2D_and_num(const vector<array2D> &vector_array, const double &num)
{
	int vector_size = vector_array.size();
	int array_col = vector_array.at(0).size();
	int array_row = vector_array.at(0).at(0).size();

	vector<array2D> vector_array_sum = vector_array;

	for (int i = 0; i < vector_size; i++)
	{
		for (int j = 0; j < array_col; j++)
		{
			for (int k = 0; k < array_row; k++)
			{
				vector_array_sum.at(i).at(j).at(k) = vector_array.at(i).at(j).at(k) + num;
			}
		}
	}

	return vector_array_sum;
}


void flip_xy_array2D(array2D &array)
{
	for (int i = 0; i < array.size(); i++)
	{
		reverse(array.at(i).begin(), array.at(i).end());
	}

	reverse(array.begin(), array.end());
}


array2D get_specific_size_array2D_from_specific_position(const array2D &X, int size_col, int size_row, int pos_col, int pos_row)
{
	if (size_col + pos_col > X.size() || size_row + pos_row > X.at(0).size())
	{
		cout << "pos and size have exceed range!" << endl << "get_specific_size_array2D_from_specific_position() has stopped!" << endl;
	}

	array2D X_patch(size_col);

	for (int i = 0; i < size_col; i++)
	{
		for (int j = 0; j <  size_row; j++)
		{
			X_patch.at(i).push_back(X.at(pos_col + i).at(pos_row + j));
		}
	}

	return X_patch;
}


array2D get_A_dot_product_B_array2D(const array2D &array_A, const array2D &array_B)
{
	int col_A = array_A.size();
	int row_A = array_A.at(0).size();
	int col_B = array_B.size();
	int row_B = array_B.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "get_A_dot_product_B_array2D() has stopped!" << endl;
		return array_A;
	}

	array2D array_AB = array_A;

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			array_AB.at(i).at(j) = array_A.at(i).at(j) * array_B.at(i).at(j);
		}
	}

	return array_AB;
}


double sum_of_array2D(const array2D &array)
{
	int col = array.size();
	int row = array.at(0).size();

	double sum_ret = 0;

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			sum_ret += array.at(i).at(j);
		}
	}

	return sum_ret;
}


vector<array2D> equal_interval_sampling_vector_array2D(const vector<array2D> &vector_array, const int sample_num)
{
	int vector_size = vector_array.size();
	int col = vector_array.at(0).size();
	int row = vector_array.at(0).at(0).size();

	int new_col = int((col + (sample_num - 1)) / sample_num); // int(3/2) = 1
	int new_row = int((row + (sample_num - 1)) / sample_num);
	if (new_col <= 0 || new_row <= 0)
	{
		cout << "array size is smaller than sample size!" << endl << "equal_interval_sampling_vector_array2D() stopped!" << endl;
		return vector_array;
	}

	vector<array2D> sampled_vector_array = create_vector_array2D(vector_size, new_col, new_row, 0);

	for (int i = 0; i < vector_size; i++)
	{
		for (int j = 0; j < new_col; j++)
		{
			for (int k = 0; k < new_row; k++)
			{
				sampled_vector_array.at(i).at(j).at(k) = vector_array.at(i).at(j * sample_num).at(k * sample_num);
			}
		}
	}

	return sampled_vector_array;
}


vector<array2D> get_A_dot_product_B_vector_array2D(const vector<array2D> &vector_array_A, const vector<array2D> &vector_array_B)
{
	int vector_size_A = vector_array_A.size();
	int col_A = vector_array_A.at(0).size();
	int row_A = vector_array_A.at(0).at(0).size();
	int vector_size_B = vector_array_B.size();
	int col_B = vector_array_B.at(0).size();
	int row_B = vector_array_B.at(0).at(0).size();

	if (vector_size_A != vector_size_B || col_A != col_B || row_A != row_B)
	{
		cout << "vector_array_A size is not same as vector_array_B size!" << endl << "get_A_dot_product_B_vector_array2D() has stopped!" << endl;
		vector<array2D> temp;
		return temp;
	}

	vector<array2D> AB_array2D;

	AB_array2D.resize(vector_size_A);

	for (int i = 0; i < vector_size_A; i++)
	{
		AB_array2D.at(i) = get_A_dot_product_B_array2D(vector_array_A.at(i), vector_array_B.at(i));
	}

	return AB_array2D;
}

