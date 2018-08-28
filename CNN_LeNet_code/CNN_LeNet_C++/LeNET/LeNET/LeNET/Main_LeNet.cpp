// 主程序main入口
#include <iostream>
#include <Main_LeNet.h>
#include <maths.h>
#include <CNN.h>
#include <string>


void test()
{
	Mat train_x1 = imread("file/2.bmp", 0);//读取灰度图
										   // normalize 归一化， 由0~255的uchar类型变为0~1的double类型
	train_x1.convertTo(train_x1, CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_32FC3为要转化的类型

	Array2Dd xxx, yyy;

	xxx.create(3, 2, 2);
	yyy.create(3, 2, 2.2);

	Array3Dd zzz;
	zzz.push_back(xxx);
	zzz.push_back(yyy);

	zzz.at(0).print();
	zzz.at(1).print();

	return;
}


int main()
{
	test();

	/*

	// ****************************** 创建训练集 ***************************************************** //

	// 加载训练集的样本图片
	vector<Mat> train_x_Mat;
	// 文件读取地址
	string file_addr = "../../../MnistData/TrainImg";
	// 批量读取图片
	read_batch_images(file_addr, "bmp", 1, 1000, train_x_Mat);
	// 转为灰度图
	images_convert_to_64FC1(train_x_Mat);
	// 由Mat图片格式转为CNN的array2D格式
	vector<array2D> train_x = vector_image_64FC1_to_vector_array2D(train_x_Mat);
	// 归一化
	normalize_vector_array2D_from_0_to_1(train_x);
	// 显示所读取的图片，即将要喂给CNN的训练数据
	vector_array2D_show_one_window("Multiple Images", train_x, CvSize(26, 18), CvSize(32, 32), 700);

	// 加载训练集的样本标签
	vector<vector<double>> train_y;
	set_target_class_one2ten(train_y,1000);
	show_vector_vector_double_as_image_64FC1(train_y, 700);

	// ****************************** 创建测试集 ***************************************************** //

	// 创建测试集的样本图片
	vector<array2D> test_x = train_x;

	// 创建测试集的样本标签
	vector<vector<double>> test_y = train_y;

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
	float alpha = 2;// 学习率[0.1,3]
	float eta = 0.5f;// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
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

	// 绘制出均方误差历史曲线
	show_curve_image(get_vector_double_n2m(0, LeNet.get_epochs()-1), LeNet.get_ERR(), 25, 1000);

	//*/
	return 0;
}



/*
Mat train_x1 = imread("file/2.bmp",0);//读取灰度图
// normalize 归一化， 由0~255的uchar类型变为0~1的double类型
train_x1.convertTo(train_x1, CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_32FC3为要转化的类型
// 显示图片
imshow("图片", train_x1);
// 等待1000ms后窗口自动关闭
waitKey(2000);
// 把图片以矩阵的形式显示出来，用于查看图片每一像素的值。
show_image_64FC1_as_matrix_double(train_x1);

double train_y[10][1000] = { 0 };
set_target_class_one2ten(train_y);
// 以图片的形式把矩阵显示出来
show_matrix_double_as_image_64FC1(train_y[0], 10, 1000, 6000);
*/
