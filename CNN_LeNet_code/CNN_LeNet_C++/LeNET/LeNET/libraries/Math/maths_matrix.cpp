#include <maths_matrix.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

using namespace std;
using namespace cv;


// 把图片以矩阵的形式显示出来，用于查看图片每一像素的值。
void show_image_64FC1_as_matrix_double(const Mat &img)
{
	int row, col;
	row = img.rows;
	col = img.cols;

	//为行指针分配空间 
	double **arr = new double *[row];
	for (int i = 0; i < row; i++)
		arr[i] = new double[col];//为每行分配空间（每行中有col个元素） 

	for (int i = 0; i < row; i++)
	{
		const double* pData = img.ptr<double>(i);	//第i+1行的所有元素
		for (int j = 0; j < col; j++)
		{
			arr[i][j] = pData[j];
		}
	}

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{

			cout.setf(ios::left); // 设置对齐方式
			cout.width(8); //设置输出宽度
			cout.fill('0'); //将多余的空格用0填充
			cout << arr[i][j] << ' ';
		}
		cout << endl;
	}

	delete[] arr;
}
