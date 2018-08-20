//
#include <iostream>
#include <Main_LeNet.h>
#include <maths.h>
#include <CNN.h>

using namespace std;

int main()
{
	// load 加载图片
	Mat train_x = imread("file/2.bmp",0);//读取灰度图

	vector<Mat> imgs(2);
	imgs[0] = imread("file/1.bmp", 0);
	imgs[1] = imread("file/3.bmp", 0);

	imgs[0].convertTo(imgs[0], CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_64FC1为要转化的类型
	imgs[1].convertTo(imgs[1], CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_64FC1为要转化的类型

	multi_image_64FC1_putin_one_window("Multiple Images", imgs, CvSize(1, 2), CvSize(32, 32), 5000);

	//return 0;

	// normalize 归一化， 由0~255的uchar类型变为0~1的double类型
	train_x.convertTo(train_x, CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_32FC3为要转化的类型

	// 显示图片   
	imshow("图片", train_x);
	// 等待1000ms后窗口自动关闭    
	waitKey(2000);

	return 0;

	// 把图片以矩阵的形式显示出来，用于查看图片每一像素的值。
	show_image_64FC1_as_matrix_double(train_x);

	double train_y[10][1000] = { 0 };
	set_target_class_one2ten(train_y);
	// 以图片的形式把矩阵显示出来
	show_matrix_double_as_image_64FC1(train_y[0], 10, 1000, 6000);

	return 0;
	

	// 定义初始化参数
	float alpha = 2;// 学习率[0.1,3]
	float eta = 0.5f;// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
	int batchsize = 10;// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
	int epochs = 25;// 训练集整体迭代次数

	// 初始化CNN
	//CNN LeNet(alpha, eta, batchsize, epochs);


	return 0;
}


void set_target_class_one2ten(double target_class[][1000])
{
	int i, j;
	int segment_size = 1000 / 10;
	for (i = 0; i <= 9; i++)
	{
		for (j = i * segment_size; j < (i + 1) * segment_size; j++)
		{
			target_class[i][j] = 1.0;
		}
	}
}

