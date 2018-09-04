#pragma once

// opencv
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream>
#include <vector>
#include <string>
#include <maths.h>
#include <Array.h>


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

	// 当前层的输出（非全连接层输出，全连接层输出为X_vector）
	vector<Array3Dd> X;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[_batchsize, iSizePic[0], iSizePic[1]]

	// 前一层通道对当前层通道的卷积核
	vector<vector<Array2Dd>> Ker;// Ker[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

	// 前一层通道对当前层通道的卷积核的偏置
	vector<vector<Array2Dd>> Ker_delta;// Ker_delta[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

	// 当前层与上一层的连接权值
	Array2Dd W;// 只针对全连接层，其它层此参数无意义
			   // 注意是W[I列][J行],I为当前层全连接输入个数，J为当前层数目
			   // 这么理解：该层有J个输出，那当然有J列了，每个输出有I个来自前层的输入，那就有I行了

	// ？？？
	Array2Dd W_delta;// 只针对全连接层，其它层此参数无意义
					 // 注意是W_delta[I列][J行],I为当前层全连接输入个数，J为当前层数目

	// 当前层输出通道的加性偏置
	vector<double> B;

	// ？？？
	vector<double> B_delta;

	// 当前层输出通道的乘性偏置
	vector<double> Beta;// 只针对下采样层，其它层此参数无意义

	// ？？？
	vector<double> Beta_delta;// 只针对下采样层，其它层此参数无意义

	// 下采样的输入
	vector<Array3Dd> X_down;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[列_batchsize, 行iChannel*iSizePic[0]*iSizePic[1]]
							// 只针对下采样层，其它层此参数无意义

	// 全连接层的输出，或者下一层为全连接层的当前层所有输出图组合成的一个向量
	Array2Dd X_vector;// 注意是_batchsize幅输入输出同时处理，所以不是一维向量，而是2D，维度为[列_batchsize, 行iChannel]
					// 只针对全连接层的输出，以及下一层为全连接层的非全连接层（比如LeNet的第二个降采样层）。其它层此参数无意义

	// 当前层的灵敏度(残差)，即当前层的输入对误差的偏导
	vector<Array3Dd> Delta;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[_batchsize, iSizePic[0], iSizePic[1]]
						   // 注意只针对非全连接层

	// 当前层的灵敏度(残差)，即当前层的输入对误差的偏导
	Array2Dd Delta_vec;// 虽是一维向量，但是有_batchsize列，所以是二维的。
					   // 注意只针对全连接层，以及下层为全连接层的降采样层、输入层、卷积层等有方块图输出的层

} Layer;


class CNN
{
public:

	// 初始化CNN类
	CNN(const vector<Layer> &layers, float alpha, float eta, int batchsize, int epochs, activation_function_type activ_func_type, down_sample_type down_samp_type);
	
	// CNN网络，训练
	void train(const Array3Dd &train_x, const Array2Dd &train_y);

	// CNN网络，测试，返回错误率
	double test(const Array3Dd &test_x, const Array2Dd &test_y);


	////////////////////// 非主要函数 /////////////////////////////////////////////

	// 返回迭代次数
	int get_epochs() { return _epochs; }

	// 返回历次迭代的均方误差
	vector<double> get_ERR() { return _ERR; }

private:
	
	// 依据网络结构设置CNN.layers, 初始化一个CNN网络
	void init();

	// CNN网络，正向计算(批处理算法,核心是convn用法,和输出层批量映射)
	void feed_forward(const Array3Dd &train_x);

	// CNN网络，反向传播(批处理算法)
	void back_propagation(const Array2Dd &train_y);

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

	// 激活函数类型
	activation_function_type _activation_func_type;

	// 降采样（池化）类型
	down_sample_type _down_sample_type;

	// 历次迭代的均方误差
	vector<double> _ERR;

	// 当前轮的当前批次的均方误差
	double _err;

	// 神经网络的输出，就是最后一层网络的输出图
	Array2Dd _Y;// 注意是_batchsize幅输入输出同时处理，所以是2D，维度为[列_batchsize, 行iChannel（即10）]
};
