#include "maxcurvature.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main()
{
	cv::Mat finger = cv::imread("finger.png", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat mask = cv::Mat::ones(finger.size(), CV_8U);         // Locus space

	cv::Mat result;
	MaxCurvature(finger, result, mask, 8);

	double min, max;
	cv::minMaxIdx(result, &min, &max);
	std::cout << max << std::endl;
	result.convertTo(result, CV_8U, 255.0/max*20);
	cv::minMaxIdx(result, &min, &max);
	std::cout << max << std::endl;
	cv::imwrite("result.png", result);
}
