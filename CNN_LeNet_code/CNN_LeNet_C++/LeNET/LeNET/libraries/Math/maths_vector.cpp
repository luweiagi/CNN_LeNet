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
vector<double> get_vector_double_n2m(int n, int m)
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
array_2D_double rand_array_2D_double(int col, int row, double minimum, double maximum)
{
	int i, j;

	array_2D_double array(col);

	for (i = 0; i < col; i++)
	{
		array[i].resize(row);
	}

	for (i = 0; i < array.size(); i++)
	{
		for (j = 0; j < array[0].size(); j++)
		{
			array[i][j] = minimum + maximum * rand() / RAND_MAX;
		}
	}

	return array;
}


// 注意：col 是该矩阵的列， row是该矩阵的行
array_2D_double zero_array_2D_double(int col, int row)
{
	return create_array_2D_double(col, row, 0);
}


// 注意：col 是该矩阵的列， row是该矩阵的行
array_2D_double create_array_2D_double(int col, int row, double value)
{
	if (col <= 0 || row <= 0)
	{
		cout << "size < 0 !" << endl << "create_array_2D_double( ) stoped!" << endl;
	}

	vector<double> vector_temp;
	vector_temp.assign(row, value);

	array_2D_double array_2D_ret;
	array_2D_ret.assign(col, vector_temp);

	return array_2D_ret;
}


void print_array_2D_double(const array_2D_double &array)
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


array_2D_double get_zero_array_2D_double_same_size_as(const array_2D_double &array)
{
	int i, j;

	array_2D_double array_ret = array;

	for (i = 0; i < array_ret.size(); i++)
	{
		array_ret[i].assign(array[i].size(), 0);
	}

	return array_ret;
}


// 把图片转为array_2D_double
array_2D_double image_64FC1_to_array_2D_double(const Mat &img)
{
	int row, col, i;
	row = img.rows;
	col = img.cols;

	// 初始化array_ret的大小尺寸
	array_2D_double array_ret(col);
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


vector<array_2D_double> vector_image_64FC1_to_vector_array_2D_double(vector<Mat> &vector_img)
{
	int size = vector_img.size();

	vector<array_2D_double> vector_array(size);

	for (int i = 0; i < size; i++)
	{
		vector_array.at(i) = image_64FC1_to_array_2D_double(vector_img.at(i));
	}

	return vector_array;
}


void normalize_array_2D_double_from_0_to_1(array_2D_double &array)
{
	// find min and max;
	double min =  100000;
	double max = -100000;

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


void normalize_vector_array_2D_double_from_0_to_1(vector<array_2D_double> &vector_array)
{
	for (int i = 0; i < vector_array.size(); i++)
	{
		normalize_array_2D_double_from_0_to_1(vector_array.at(i));
	}
}


void zero_vector_array_2D_double(vector<array_2D_double> &vector_array)
{
	array_2D_double array_2D_zero = get_zero_array_2D_double_same_size_as(vector_array.at(1));
	vector_array.assign(vector_array.size(), array_2D_zero);
}


vector<array_2D_double> get_zero_vector_array_2D_double_same_size_as(const vector<array_2D_double> &vector_array)
{
	vector<array_2D_double>array_ret = vector_array;

	zero_vector_array_2D_double(array_ret);

	return array_ret;
}


vector<array_2D_double> create_vector_array_2D_double(int vector_size, int array_col, int array_row, double value)
{
	if (vector_size <= 0 || array_col <= 0 || array_row <= 0)
	{
		cout << "size < 0 !" << endl << "create_vector_array_2D_double( ) stoped!" << endl;
	}

	vector<double> vector_temp;
	vector_temp.assign(array_row, value);

	array_2D_double array_2D_temp;
	array_2D_temp.assign(array_col, vector_temp);

	vector<array_2D_double> array_3D_ret;
	array_3D_ret.assign(vector_size, array_2D_temp);

	return array_3D_ret;
}


void add_B_to_A_vector_array_2D_double(vector<array_2D_double> &vector_array_A, const vector<array_2D_double> &vector_array_B)
{
	int vector_size = vector_array_A.size();
	int array_col = vector_array_A.at(0).size();
	int array_row = vector_array_A.at(0).at(0).size();

	int vector_size_B = vector_array_B.size();
	int array_col_B = vector_array_B.at(0).size();
	int array_row_B = vector_array_B.at(0).at(0).size();

	if (vector_size != vector_size_B || array_col != array_col_B || array_row != array_row_B)
	{
		cout << "vector_array_A size is not same to vector_array_B size! " << endl << "add_vector_array_2D_double() stoped!" << endl;
		return;
	}

	for (int i = 0; i < vector_size; i++)
	{
		for (int j = 0; j < array_col; j++)
		{
			for (int k = 0; k < array_row; k++)
			{
				vector_array_A.at(i).at(j).at(k) += vector_array_B.at(i).at(j).at(k);
			}
		}
	}
}


void flip_xy_array_2D_double(array_2D_double &array)
{
	for (int i = 0; i < array.size(); i++)
	{
		reverse(array.at(i).begin(), array.at(i).end());
	}

	reverse(array.begin(), array.end());
}


array_2D_double get_specific_size_array_2D_double_from_specific_position(const array_2D_double &X, int size_col, int size_row, int pos_col, int pos_row)
{
	if (size_col + pos_col > X.size() || size_row + pos_row > X.at(0).size())
	{
		cout << "pos and size have exceed range!" << endl << "get_specific_size_array_2D_double_from_specific_position() has stopped!" << endl;
	}

	array_2D_double X_patch(size_col);

	for (int i = 0; i < size_col; i++)
	{
		for (int j = 0; j <  size_row; j++)
		{
			X_patch.at(i).push_back(X.at(pos_col + i).at(pos_row + j));
		}
	}

	return X_patch;
}


array_2D_double get_A_dot_product_B_array_2D_double(const array_2D_double &array_A, const array_2D_double &array_B)
{
	int col_A = array_A.size();
	int row_A = array_A.at(0).size();
	int col_B = array_B.size();
	int row_B = array_B.at(0).size();

	if (col_A != col_B || row_A != row_B)
	{
		cout << "array_A size is not same as arrow_B size!" << endl << "get_A_dot_product_B_array_2D_double() has stopped!" << endl;
		return array_A;
	}

	array_2D_double array_AB = array_A;

	for (int i = 0; i < col_A; i++)
	{
		for (int j = 0; j < row_A; j++)
		{
			array_AB.at(i).at(j) = array_A.at(i).at(j) * array_B.at(i).at(j);
		}
	}

	return array_AB;
}


double sum_of_array_2D_double(const array_2D_double &array)
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
