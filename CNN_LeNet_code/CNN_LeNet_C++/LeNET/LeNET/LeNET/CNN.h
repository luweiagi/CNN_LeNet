#pragma once

// opencv
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream>
#include <vector>
#include <string>
#include <maths.h>

using namespace std;
using namespace cv;

typedef struct{
	// 当前层的类别
	char type;// 输入层：i；卷积层：c；降采样层：s；全连接层：f

	// 当前层的通道数目
	int iChannel;

	// 降采样率
	int iSample;// 只针对降采样层，其它层此参数无意义

	// 当前层的输入图片大小
	int iSizePic[2];

	// 当前层的卷积核大小[row col]
	int iSizeKer;// 只针对卷积层，其它层此参数无意义

	// 当前层的输出
	vector<array_3D_double> X;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[_batchsize, iSizePic[0], iSizePic[1]]

	// 前一层通道对当前层通道的卷积核
	vector<vector<array_2D_double>> Ker;// Ker[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

	// 前一层通道对当前层通道的卷积核的偏置
	vector<vector<array_2D_double>> Ker_delta;// Ker_delta[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

	// 当前层与上一层的连接权值
	array_2D_double W;// 只针对全连接层，其它层此参数无意义

	// ？？？
	array_2D_double W_delta;// 只针对全连接层，其它层此参数无意义

	// 当前层输出通道的加性偏置
	vector<double> B;

	// ？？？
	vector<double> B_delta;

	// 当前层输出通道的乘性偏置
	vector<double> Beta;// 只针对下采样层，其它层此参数无意义

	// ？？？
	vector<double> Beta_delta;// 只针对下采样层，其它层此参数无意义

} Layer;


class CNN
{
public:

	// 初始化CNN类
	CNN(vector<Layer> layers, float alpha, float eta, int batchsize, int epochs)
		:_layers(layers), _alpha(alpha), _eta(eta), _batchsize(batchsize), _epochs(epochs)
	{
		// 依据网络结构设置CNN.layers, 初始化一个CNN网络
		init();

		_ERR.assign(_epochs, 0);// 将历次迭代的均方误差初始化为0
		_err = 0;// 将当前轮的当前批次的均方误差初始化为0

		cout << "CNN has initialised!" << endl;
	}
	
	// CNN网络，训练
	void train(const vector<array_2D_double> &train_x, const vector<vector<double>> &train_y);

	// CNN网络，测试，返回错误率
	double test(const vector<array_2D_double> &test_x, const vector<vector<double>> &test_y);


	////////////////////// 非主要函数 /////////////////////////////////////////////

	// 返回迭代次数
	int get_epochs() { return _epochs; }

	// 返回历次迭代的均方误差
	vector<double> get_ERR() { return _ERR; }

private:
	
	// 依据网络结构设置CNN.layers, 初始化一个CNN网络
	void init();

	// CNN网络，正向计算(批处理算法,核心是convn用法,和输出层批量映射)
	void feed_forward(const vector<array_2D_double> &train_x);

	// CNN网络，反向传播(批处理算法)
	void back_propagation(const vector<vector<double>> &train_y);

	// CNN网络，卷积层和输出层的权值更新(附加惯性项)
	void update(void);

	vector<Layer> _layers;

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

	// 当前轮的当前批次的均方误差
	double _err;
};
