#ifndef MAX_CURVATURE_H
#define MAX_CURVATURE_H

#include <opencv2/imgproc/imgproc.hpp>

void MaxCurvature(cv::InputArray src, cv::OutputArray dst, cv::InputArray mask, double sigma);

#endif // MAX_CURVATURE_H
