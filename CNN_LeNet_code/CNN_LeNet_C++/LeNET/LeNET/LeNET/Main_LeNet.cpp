// 主程序main入口
#include <iostream>
#include <Main_LeNet.h>
#include <maths.h>
#include <CNN.h>
#include <string>


void test()
{

}

int main()
{
	test();

	//*

	// ****************************** 创建训练集 ***************************************************** //

	// 加载训练集的样本图片
	vector<Mat> train_x;
	string file_addr = "../../../MnistData/TrainImg";
	read_batch_images(file_addr, "bmp", 1, 1000, train_x);
	images_convert_to_64FC1(train_x);
	multi_images_64FC1_show_one_window("Multiple Images", train_x, CvSize(26, 18), CvSize(32, 32), 700);
	
	// 加载训练集的样本标签
	vector<vector<double>> train_y;
	set_target_class_one2ten(train_y,1000);
	show_vector_vector_double_as_image_64FC1(train_y, 700);

	// ****************************** 创建测试集 ***************************************************** //

	// 创建测试集的样本图片
	vector<Mat> test_x = train_x;

	// 创建测试集的样本标签
	vector<vector<double>> test_y;
	test_y.assign(train_y.begin(), train_y.end());

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

	// 依据网络结构设置CNN.layers，初始化一个CNN网络
	CNN LeNet(layers, alpha, eta, batchsize, epochs);

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


void set_target_class_one2ten(vector<vector<double>> &target_class, int length)
{
	int segment_size = length / 10;
	int i, j;
	vector<double> one_hot;

	for (i = 0; i <= 8; i++)
	{
		one_hot.assign(10, 0);
		one_hot.at(i) = 1;

		for (j = 0; j < segment_size; j++)
		{
			target_class.push_back(one_hot);
		}
	}

	// 防止length被10除不开，比如length=1003，则最后的类别10为从900到1003，而不是900到1000。
	one_hot.assign(10, 0);
	one_hot.at(i) = 1;
	for (j = i * segment_size; j < length; j++)
	{
		target_class.push_back(one_hot);
	}
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
