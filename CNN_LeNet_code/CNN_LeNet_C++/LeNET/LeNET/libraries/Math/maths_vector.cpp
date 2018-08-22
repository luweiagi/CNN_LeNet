#include <maths_vector.h>
#include <iostream>
#include <time.h>


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
array_double rand_array_double(int col, int row, double minimum, double maximum)
{
	int i, j;

	array_double array(col);

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
array_double zero_array_double(int col, int row)
{
	int i, j;

	array_double array(col);

	for (i = 0; i < col; i++)
	{
		array[i].resize(row);
	}

	for (i = 0; i < array.size(); i++)
	{
		for (j = 0; j < array[0].size(); j++)
		{
			array[i][j] = 0;
		}
	}

	return array;
}


// 注意：col 是该矩阵的列， row是该矩阵的行
array_double array_double_with_assigned_value(int col, int row, double value)
{
	int i, j;

	array_double array(col);

	for (i = 0; i < col; i++)
	{
		array[i].resize(row);
	}

	for (i = 0; i < array.size(); i++)
	{
		for (j = 0; j < array[0].size(); j++)
		{
			array[i][j] = value;
		}
	}

	return array;
}


void print_array_double(const array_double &array)
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


array_double get_zero_array_double_same_size_as(const array_double &array)
{
	int i, j;

	array_double array_ret = array;

	for (i = 0; i < array_ret.size(); i++)
	{
		array_ret[i].assign(array[i].size(), 0);
	}

	return array_ret;
}



