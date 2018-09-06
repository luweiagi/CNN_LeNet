#include <maths_vector.h>
#include <iostream>
#include <time.h>
#include<algorithm>


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
