
#include <maths.h>
#include <maths_image.h>


Mat mat2x2_to_Mat(uchar **array, int row, int col)
{
	Mat img(row, col, CV_8UC1);
	uchar *ptmp = NULL;
	for (int i = 0; i <row; ++i)
	{
		ptmp = img.ptr<uchar>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = array[i][j];
		}
	}

	return img;
}


void show_mat2x2_as_image(uchar **array, int row, int col, int time_msec)
{
	Mat image = mat2x2_to_Mat(array, row, col);

	// 显示图片   
	imshow("图片", image);
	// 等待time_msec后窗口自动关闭    
	waitKey(time_msec);
}
