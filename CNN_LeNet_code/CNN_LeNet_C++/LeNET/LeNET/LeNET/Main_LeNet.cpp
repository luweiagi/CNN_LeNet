// 主程序main入口
#include <iostream>
#include <Main_LeNet.h>
#include <maths.h>
#include <CNN.h>
#include <string>
#include <time.h>


void test()
{
	;
}


int main()
{
	//test();

	//*

	// ****************************** 创建训练集 ***************************************************** //

	clock_t tic_LeNet = clock();// LeNet运行计时

	// 加载训练集的样本图片
	vector<Mat> train_x_Mat;
	// 文件读取地址
	string file_addr = "../../../MnistData/TrainImg";
	// 批量读取图片
	read_batch_images(file_addr, "bmp", 1, 1000, train_x_Mat);
	// 转为灰度图
	images_convert_to_64FC1(train_x_Mat);

	// 由Mat图片格式转为CNN所用的Array3Dd格式
	Array3Dd train_x(train_x_Mat);
	// 归一化
	train_x.normalize();
	// 显示所读取的图片，即将要喂给CNN的训练数据
	train_x.show_specified_images_64FC1("Multiple Images", CvSize(26, 18), CvSize(32, 32), 1000);

	// 加载训练集的样本标签
	Array2Dd train_y;
	train_y.class_0_to_9(1000);
	train_y.show_image_64FC1(700);

	// ****************************** 创建测试集 ***************************************************** //

	// 创建测试集的样本图片
	Array3Dd test_x = train_x;

	// 创建测试集的样本标签
	Array2Dd test_y = train_y;

	// ****************************** 初始化CNN ****************************************************** //

	// CNN网络结构设置
	vector<Layer> layers;

	Layer input_layer_1;// 第一层：输入层
	input_layer_1.type = 'i'; input_layer_1.iChannel = 1; input_layer_1.iSizePic[0] = 32; input_layer_1.iSizePic[1] = 32;
	layers.push_back(input_layer_1);

	Layer convolutional_layer_2;// 第二层：卷积层
	convolutional_layer_2.type = 'c'; convolutional_layer_2.iChannel = 2; convolutional_layer_2.iSizeKer = 5;
	layers.push_back(convolutional_layer_2);

	Layer subsampling_layer_3;// 第三层：降采样层
	subsampling_layer_3.type = 's'; subsampling_layer_3.iSample = 2;
	layers.push_back(subsampling_layer_3);

	Layer convolutional_layer_4;// 第四层：卷积层
	convolutional_layer_4.type = 'c'; convolutional_layer_4.iChannel = 4; convolutional_layer_4.iSizeKer = 5;
	layers.push_back(convolutional_layer_4);

	Layer subsampling_layer_5;// 第五层：降采样层
	subsampling_layer_5.type = 's'; subsampling_layer_5.iSample = 2;
	layers.push_back(subsampling_layer_5);

	Layer fully_connected_layer_6;// 第六层：全连接层
	fully_connected_layer_6.type = 'f'; fully_connected_layer_6.iChannel = 120;
	layers.push_back(fully_connected_layer_6);

	Layer fully_connected_layer_7;// 第七层：全连接层
	fully_connected_layer_7.type = 'f'; fully_connected_layer_7.iChannel = 84;
	layers.push_back(fully_connected_layer_7);

	Layer fully_connected_layer_8;// 第八层：全连接层（输出层）
	fully_connected_layer_8.type = 'f'; fully_connected_layer_8.iChannel = 10;
	layers.push_back(fully_connected_layer_8);

	// 定义初始化参数
	double alpha = 2;// 学习率[0.1,3]
	double eta = 0.5f;// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
	int batchsize = 10;// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
	int epochs = 25;// 训练集整体迭代次数
	activation_function_type activ_func_type = SoftMax;// 激活函数类型
	down_sample_type down_samp_type = MeanPooling;// 降采样（池化）类型

	// 依据网络结构设置CNN.layers，初始化一个CNN网络
	CNN LeNet(layers, alpha, eta, batchsize, epochs, activ_func_type, down_samp_type);

	// ****************************** CNN训练 ******************************************************** //

	// CNN网络训练
	LeNet.train(train_x, train_y);

	// ****************************** CNN测试 ******************************************************** //

	// CNN网络测试，用测试样本来测试
	double error_rate = LeNet.test(test_x, test_y);

	// 测试结果输出
	cout << "error rate is ： " << error_rate * 100 << "%" << endl;

	cout << "error history is ： " << endl;
	print(LeNet.get_ERR());

	// 绘制出均方误差历史曲线
	show_curve_image(get_vector_n2m(0, LeNet.get_epochs()-1), LeNet.get_ERR() * 50.0, 25, 30000);

	clock_t toc_LeNet = clock();// LeNet运行计时

	cout << "LeNet total time: " << (double)(toc_LeNet - tic_LeNet) / 1000 << " seconds" << endl;
	//*/
	return 0;
}
