#pragma once

// opencv
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

template <typename T>

class CNN
{
public:

	CNN(float alpha, float eta, int batchsize, int epochs)
		:alpha(alpha), eta(eta), batchsize(batchsize), epochs(epochs)
	{

	}
	
	//train();

	// 学习率[0.1,3]
	float alpha;

	// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
	float eta;

	// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
	int batchsize;

	// 训练集整体迭代次数
	int epochs;

private:
	
};
