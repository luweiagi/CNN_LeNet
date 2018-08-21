#pragma once

// opencv
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

class CNN
{
public:

	CNN(float alpha, float eta, int batchsize, int epochs)
		:_alpha(alpha), _eta(eta), _batchsize(batchsize), _epochs(epochs)
	{
		cout << "already init CNN" << endl;

		_ERR.assign(_epochs, 0);
	}
	
	// 训练
	void train(vector<Mat> train_x, vector<vector<double>> train_y);

	// 返回错误率
	double test(vector<Mat> test_x, vector<vector<double>> test_y);


	//  不重要的函数

	// 返回迭代次数
	int get_epochs() { return _epochs; }

	// 返回历次迭代的均方误差
	vector<double> get_ERR() { return _ERR; }

private:
	

	// 学习率[0.1,3]
	float _alpha;

	// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
	float _eta;

	// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
	int _batchsize;

	// 训练集整体迭代次数
	int _epochs;

	// 历次迭代的均方误差
	vector<double> _ERR;

};
