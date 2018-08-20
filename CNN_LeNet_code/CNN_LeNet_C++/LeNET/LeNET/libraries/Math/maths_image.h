#pragma once

#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

// 不能出现：using namespace cv;
using namespace std;

cv::Mat matrix_double_to_Mat_64FC1(double *array, int row, int col);

void show_matrix_double_as_image_64FC1(double *array, int row, int col, int time_msec);

void multi_image_64FC1_putin_one_window(const std::string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size, int time_msec);

void multi_image_8UC3_putin_one_window(const std::string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size, int time_msec);

