#include <iostream>
#include <maths_image.h>
#include <windows.h>
#include <vector_array.h>

using namespace std;


// 将多幅单通道灰度图输出到一个图里，类似matlab的subplot
void vector_Mat_64FC1_show_one_window(const std::string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size, int time_msec)
{
	//Reference : http://blog.csdn.net/yangyangyang20092010/article/details/21740373

	//============= Usage ==============//
	//vector<Mat> imgs(4);
	//imgs[0] = imread("F:\\SA2014.jpg");
	//imgs[1] = imread("F:\\SA2014.jpg");
	//imgs[2] = imread("F:\\SA2014.jpg");
	//imgs[3] = imread("F:\\SA2014.jpg");
	//MultiImage_OneWin("T", imgs, cvSize(2, 2), cvSize(400, 280));

	int Img_Num = (int)SrcImg_V.size();
	if (Img_Num < SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot num Setting is too large !" << endl;
		exit(0);
	}

	// select image based on the SubPlot size
	vector<cv::Mat> SrcImg_V_Selected;
	SrcImg_V_Selected.assign(SrcImg_V.begin(), SrcImg_V.begin() + SubPlot.width * SubPlot.height);

	//Window's image
	cv::Mat Disp_Img;
	//Width of source image
	CvSize Img_OrigSize = cvSize(SrcImg_V_Selected[0].cols, SrcImg_V_Selected[0].rows);
	//================ Set the width for displayed image =======================//
	//Width vs height ratio of source image
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//================ Check Image numbers with Subplot layout ======================//
	int Img_Selected_Num = (int)SrcImg_V_Selected.size();
	if (Img_Selected_Num > SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot Setting is too small !" << endl;
		exit(0);
	}
	//=================== Blank setting ====================//
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap = cvSize(15, 15);
	//=================== Size for Window ===================//
	// CV_8UC3:8位三通道彩图0~256   CV_64FC1：64位单通道灰度图0~1
	Disp_Img.create(cv::Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_64FC1);
	Disp_Img.setTo(0);// Background
					  // Left top position for each image
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	// Display all images
	for (int i = 0; i < Img_Selected_Num; i++)
	{
		// Obtain the left top position
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		// Writting each to Window's Image
		cv::Mat imgROI = Disp_Img(cv::Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		cv::resize(SrcImg_V_Selected[i], imgROI, cv::Size(ImgDisp_Size.width, ImgDisp_Size.height));
		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}
	// Get the screen size of computer
	int Scree_W = GetSystemMetrics(SM_CXSCREEN);
	int Scree_H = GetSystemMetrics(SM_CYSCREEN);
	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//Centralize the window
	cvShowImage(MultiShow_WinName.c_str(), &(IplImage(Disp_Img)));
	cvWaitKey(time_msec);
	cvDestroyWindow(MultiShow_WinName.c_str());
}


// 将多幅三通道彩图输出到一个图里，类似matlab的subplot
void vector_Mat_8UC3_show_one_window(const std::string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size, int time_msec)
{
	//Reference : http://blog.csdn.net/yangyangyang20092010/article/details/21740373

	//============= Usage ==============//
	//vector<Mat> imgs(4);
	//imgs[0] = imread("F:\\SA2014.jpg");
	//imgs[1] = imread("F:\\SA2014.jpg");
	//imgs[2] = imread("F:\\SA2014.jpg");
	//imgs[3] = imread("F:\\SA2014.jpg");
	//MultiImage_OneWin("T", imgs, cvSize(2, 2), cvSize(400, 280));

	int Img_Num = (int)SrcImg_V.size();
	if (Img_Num < SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot num Setting is too large !" << endl;
		exit(0);
	}

	// select image based on the SubPlot size
	vector<cv::Mat> SrcImg_V_Selected;
	SrcImg_V_Selected.assign(SrcImg_V.begin(), SrcImg_V.begin() + SubPlot.width * SubPlot.height);

	//Window's image
	cv::Mat Disp_Img;
	//Width of source image
	CvSize Img_OrigSize = cvSize(SrcImg_V_Selected[0].cols, SrcImg_V_Selected[0].rows);
	//================ Set the width for displayed image =======================//
	//Width vs height ratio of source image
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//================ Check Image numbers with Subplot layout ======================//
	int Img_Selected_Num = (int)SrcImg_V_Selected.size();
	if (Img_Selected_Num > SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot Setting is too small !" << endl;
		exit(0);
	}
	//=================== Blank setting ====================//
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap = cvSize(15, 15);
	//=================== Size for Window ===================//
	// CV_8UC3:8位三通道彩图0~256   CV_64FC1：64位单通道灰度图0~1
	Disp_Img.create(cv::Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_8UC3);
	Disp_Img.setTo(0);// Background
					  // Left top position for each image
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	// Display all images
	for (int i = 0; i < Img_Selected_Num; i++)
	{
		// Obtain the left top position
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		// Writting each to Window's Image
		cv::Mat imgROI = Disp_Img(cv::Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		cv::resize(SrcImg_V_Selected[i], imgROI, cv::Size(ImgDisp_Size.width, ImgDisp_Size.height));
		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}
	// Get the screen size of computer
	int Scree_W = GetSystemMetrics(SM_CXSCREEN);
	int Scree_H = GetSystemMetrics(SM_CYSCREEN);
	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//Centralize the window
	cvShowImage(MultiShow_WinName.c_str(), &(IplImage(Disp_Img)));
	cvWaitKey(time_msec);
	cvDestroyWindow(MultiShow_WinName.c_str());
}


// 将多幅array2D输出到一个图里，类似matlab的subplot。 注意，这里应当保证array2D的值介于0~1之间
void vector_array2D_show_one_window(const std::string& MultiShow_WinName, const vector<array2D>& vector_array, CvSize SubPlot, CvSize ImgMax_Size, int time_msec)
{
	//Reference : http://blog.csdn.net/yangyangyang20092010/article/details/21740373

	//============= Usage ==============//
	//vector<Mat> imgs(4);
	//imgs[0] = imread("F:\\SA2014.jpg");
	//imgs[1] = imread("F:\\SA2014.jpg");
	//imgs[2] = imread("F:\\SA2014.jpg");
	//imgs[3] = imread("F:\\SA2014.jpg");
	//MultiImage_OneWin("T", imgs, cvSize(2, 2), cvSize(400, 280));

	vector<cv::Mat> SrcImg_V = vector_array2D_to_vector_Mat_64FC1(vector_array);

	int Img_Num = (int)SrcImg_V.size();
	if (Img_Num < SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot num Setting is too large !" << endl;
		exit(0);
	}

	// select image based on the SubPlot size
	vector<cv::Mat> SrcImg_V_Selected;
	SrcImg_V_Selected.assign(SrcImg_V.begin(), SrcImg_V.begin() + SubPlot.width * SubPlot.height);

	//Window's image
	cv::Mat Disp_Img;
	//Width of source image
	CvSize Img_OrigSize = cvSize(SrcImg_V_Selected[0].cols, SrcImg_V_Selected[0].rows);
	//================ Set the width for displayed image =======================//
	//Width vs height ratio of source image
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//================ Check Image numbers with Subplot layout ======================//
	int Img_Selected_Num = (int)SrcImg_V_Selected.size();
	if (Img_Selected_Num > SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot Setting is too small !" << endl;
		exit(0);
	}
	//=================== Blank setting ====================//
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap = cvSize(15, 15);
	//=================== Size for Window ===================//
	// CV_8UC3:8位三通道彩图0~256   CV_64FC1：64位单通道灰度图0~1
	Disp_Img.create(cv::Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_64FC1);
	Disp_Img.setTo(0);// Background
					  // Left top position for each image
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	// Display all images
	for (int i = 0; i < Img_Selected_Num; i++)
	{
		// Obtain the left top position
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		// Writting each to Window's Image
		cv::Mat imgROI = Disp_Img(cv::Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		cv::resize(SrcImg_V_Selected[i], imgROI, cv::Size(ImgDisp_Size.width, ImgDisp_Size.height));
		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}
	// Get the screen size of computer
	int Scree_W = GetSystemMetrics(SM_CXSCREEN);
	int Scree_H = GetSystemMetrics(SM_CYSCREEN);
	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//Centralize the window
	cvShowImage(MultiShow_WinName.c_str(), &(IplImage(Disp_Img)));
	cvWaitKey(time_msec);
	cvDestroyWindow(MultiShow_WinName.c_str());
}
