#include <CNN.h>
#include <maths.h>
#include <time.h>


#include<iostream>
using namespace std;

void CNN::train(vector<Mat> train_x, vector<vector<double>> train_y)
{
	cout << "begin to train" << endl;

	if (train_x.size() != train_y.size())
	{
		cout << "train_x size is not same as train_y size!" << endl << "stop CNN trainning!" << endl;
		return;
	}

	int m = train_x.size();// 训练样本个数
	int numbatches = ceil(m / _batchsize);// "训练集整体迭代一次" 网络权值更新的次数

	for (int I = 0; I < _epochs; I++)
	{
		cout << "epoch " << I+1 << "/" << _epochs << endl;

		clock_t tic = clock(); //获取毫秒级数目
		



		clock_t toc = clock(); //获取毫秒级数目
		cout << "time has elapsed: " << (double)(toc - tic) / 1000 << " seconds" << endl;
	}
}


double CNN::test(vector<Mat> test_x, vector<vector<double>> test_y)
{
	cout << "begin to test" << endl;
	return 0;
}
