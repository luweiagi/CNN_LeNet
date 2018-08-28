#include <iostream>
#include <maths.h>
#include <maths_image.h>


using namespace cv;

// 此函数的array二维数组不能是new出来的！！！因为内存地址不连续，所以会越界。
Mat matrix_double_to_Mat_64FC1(double *array, int row, int col)
{
	Mat img(row, col, CV_64FC1);
	double *ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = *array++;
		}
	}
	return img;
}


Mat vector_vector_double_to_Mat_64FC1(const vector<vector<double>> &array)
{
	int row = array.at(0).size();
	int col = array.size();

	Mat img(row, col, CV_64FC1);
	double *ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = array.at(j).at(i);
		}
	}
	return img;
}


vector<Mat> vector_array2D_to_vector_Mat_64FC1(const vector<array2D> &vector_array)
{
	int size = vector_array.size();

	vector<Mat> vector_Mat_ret(size);

	for (int i = 0; i < size; i++)
	{
		vector_Mat_ret.at(i) = vector_vector_double_to_Mat_64FC1(vector_array.at(i));
	}
	
	return vector_Mat_ret;
}


// 以图片的形式把矩阵显示出来
// 调用此函数时，第一个参数必须写成array[0],而不能是array
void show_matrix_double_as_image_64FC1(double *array, int row, int col, int time_msec)
{
	Mat image = matrix_double_to_Mat_64FC1(array, row, col);

	// 显示图片   
	imshow("图片", image);
	// 等待time_msec后窗口自动关闭    
	waitKey(time_msec);
}


// 以图片的形式把vector矩阵显示出来
void show_vector_vector_double_as_image_64FC1(const vector<vector<double>> &array, int time_msec)
{
	Mat image = vector_vector_double_to_Mat_64FC1(array);

	// 显示图片   
	imshow("图片", image);
	// 等待time_msec后窗口自动关闭    
	waitKey(time_msec);

	destroyWindow("图片");
}


// 从指定文件夹内批量读取图片
void read_batch_images(string file_addr, string image_suffix, int begin_num, int end_num, vector<Mat> &data_set)
{
	for (int i = begin_num; i <= end_num; i++)
	{
		stringstream ss; // int转string
		string image_name;
		ss << i;
		ss >> image_name;
		image_name = image_name + "." + image_suffix;
		string image_addr_name = file_addr + "/" + image_name;
		cout << "reading " << image_name << " from " << file_addr << endl;

		// 读取灰度图
		Mat image = imread(image_addr_name, 0);

		data_set.push_back(image);

		if (image.data == 0)
		{
			cout << "[warning: no image!]" << endl;
		}
	}
}


void images_convert_to_64FC1(vector<Mat> &data_set)
{
	vector<Mat>::iterator it;
	for (it = data_set.begin(); it != data_set.end(); it++)
	{
		(*it).convertTo(*it, CV_64FC1, 1 / 255.0);//其中dst为目标图， CV_64FC1为要转化的类型
	}
}


/*
	scale参数用于调整输出图像内容占据整个画布的比例
*/
void show_curve_image(vector<double>data_x, vector<double>data_y, float scale, int msec)
{
	// https://blog.csdn.net/hu_guan_jie/article/details/50987520

	if (data_x.size() != data_y.size())
	{
		cout << "data_x size is not same as data_y size in show_curve_image()!" << endl;
		return;
	}

	int point_num = data_x.size();

	Mat img = Mat::zeros(800, 800, CV_8UC3);//创建画布

	vector<Point> curvePoint;//用于保存point的vector
	Point tmpPoint;

	for (int i = 0; i < point_num; ++i)
	{
		tmpPoint = cvPoint((int)data_x.at(i)*scale, (int)data_y.at(i)*scale);;
		curvePoint.push_back(tmpPoint);
	}

	vector<Point>::iterator it;
	it = curvePoint.begin();

	Point pointPre = cvPoint(curvePoint.at(1).x, 800 - curvePoint.at(1).y);//起始点
	while (it != curvePoint.end())
	{
		Point pointTmp = (*it);
		pointTmp = cvPoint(pointTmp.x, 800 - pointTmp.y);//坐标翻转
		line(img, pointPre, pointTmp, cvScalarAll(255), 4);
		pointPre = pointTmp;

		it++;
	}

	imshow("curve", img);
	waitKey(msec);
}


