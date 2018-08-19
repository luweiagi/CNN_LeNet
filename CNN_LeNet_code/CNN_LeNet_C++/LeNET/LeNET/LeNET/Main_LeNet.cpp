#include <stdio.h>
#include <iostream>
#include <Main_LeNet.h>
#include <maths.h>
#include <CNN.h>

using namespace std;

int main()
{
	/*
	// 读入一张图片
	Mat img = imread("file/lena.jpg");

	// 显示图片   
	imshow("图片", img);
	// 等待1000ms后窗口自动关闭    
	waitKey(1000);
	
	
	VideoCapture cap(0);
	Mat frame;
	while (1)
	{
		cap >> frame;
		imshow("调用摄像头", frame);
		waitKey(10);
	}
	*/

	// load 加载图片
	Mat train_x = imread("file/lena.jpg");
	uchar train_y[10][1000] = { 0 };
	set_target_class_one2ten(train_y, 1000);
	print_matrix2x2((unsigned char**)train_y, 10, 1000);
	show_mat2x2_as_image((uchar**)train_y, 10, 1000, 6000);


	// 显示图片   
	imshow("图片", train_x);
	// 等待1000ms后窗口自动关闭    
	waitKey(1000);

	// 定义初始化参数
	float alpha = 2;// 学习率[0.1,3]
	float eta = 0.5f;// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
	int batchsize = 10;// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
	int epochs = 25;// 训练集整体迭代次数

	// 初始化CNN
	//CNN LeNet(alpha, eta, batchsize, epochs);


	return 0;
}


void set_target_class_one2ten(uchar target_class[][1000], int length)
{
	int i, j;
	int segment_size = 1000 / length;
	for (i = 0; i < 9; i++)
	{
		for (j = i * segment_size; j < (i + 1) * segment_size; j++)
		{
			target_class[i][j] = 1;
		}
	}
}

