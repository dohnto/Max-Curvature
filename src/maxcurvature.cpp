#include <iostream>
#include "maxcurvature.h"

// DELETE
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>      // std::setprecision

#include <opencv2/core/types_c.h>
#include <opencv2/core/operations.hpp>
#include <cmath>
#include <algorithm>    // std::random_shuffle

void _meshgrid(const cv::Range &_xgv, const cv::Range &_ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;

	for (int i = _xgv.start; i <= _xgv.end; i++) {
		t_x.push_back(i);
	}
	for (int i = _ygv.start; i <= _ygv.end; i++) {
		t_y.push_back(i);
	}

	cv::Mat xgv(t_x);
	cv::Mat ygv(t_y);

	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

std::string type2str(int type)
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

template<class T>
void _printMatrix(cv::Mat & m, int showColumns = 7)
{
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.precision(4);

	for (int i = 0; i <= ceil(m.cols / showColumns); i++) {
		for (int y = 0; y < m.rows; ++y) {
			for (int x = 0; x < showColumns; ++x) {
				int column = x + i * showColumns;
				if (column < m.rows) {
					std::cout << "    " << m.at<T>(y, column);
				}
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

cv::Mat _conv(cv::Mat src, cv::Mat kernel)
{
	cv::Mat result;
	cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	cv::Mat flipped;
	cv::flip(kernel, flipped, 0);
	cv::filter2D(src, result, -1, flipped, anchor, 0, cv::BORDER_REPLICATE);

	return result;
}

void MaxCurvature(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, double sigma)
{
	cv::Mat src = _src.getMat().clone();

	src.convertTo(src, CV_32F, 1.0 / 255.0);

	cv::Mat mask = _mask.getMat();

	double sigma2 = std::pow(sigma, 2);
	double sigma4 = std::pow(sigma, 4);

	// Construct filter kernels
	int winsize = std::ceil(4 * sigma);
	cv::Mat1i X, Y;
	_meshgrid(cv::Range(-winsize, winsize), cv::Range(-winsize, winsize), X, Y);

	// Construct h
	cv::Mat X2, Y2;
	cv::pow(X, 2, X2);
	cv::pow(Y, 2, Y2);
	cv::Mat1f X2Y2 = X2 + Y2;

	cv::Mat expXY;
	cv::exp(-X2Y2 / (2 * sigma2), expXY);
	cv::Mat h = (1 / (2 * M_PI * std::pow(sigma, 2))) * expXY;

	// Construct hx
	cv::Mat1f Xsigma2 = -X / sigma2;
	cv::Mat hx;
	cv::multiply(h, Xsigma2, hx);

	// Construct hxx
	cv::Mat1f temp = ((X2 - sigma2) / sigma4);
	cv::Mat hxx;
	cv::multiply(h, temp, hxx);

	// Construct hy
	cv::Mat hy;
	cv::transpose(hx, hy);

	// Construct hyy
	cv::Mat hyy;
	cv::transpose(hxx, hyy);

	// Construct hxy
	cv::Mat1f XY;
	cv::multiply(X, Y, XY);
	cv::Mat hxy;
	temp = XY / sigma4;
	cv::multiply(temp, h, hxy);

	cv::Mat fx = -_conv(src, hx);                                                 // WHY FUCKING MINUS
	cv::Mat fxx = _conv(src, hxx);
	cv::Mat fy = _conv(src, hy);
	cv::Mat fyy = _conv(src, hyy);
	cv::Mat fxy = -_conv(src, hxy);                                               // WHY FUCKING MINUS

	/*  \   */
	cv::Mat f1  = 0.5 * sqrt(2.0) * (fx + fy);
	/*  /   */
	cv::Mat f2  = 0.5 * sqrt(2.0) * (fx - fy);
	/*  \\  */
	cv::Mat f11 = 0.5 * fxx + fxy + 0.5 * fyy;
	/*  //  */
	cv::Mat f22 = 0.5 * fxx - fxy + 0.5 * fyy;

	int img_h = src.size().height;
	int img_w = src.size().width;

	cv::Mat k1 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat k2 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat k3 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat k4 = cv::Mat::zeros(src.size(), CV_32F);


	for (int x = 0; x < img_w; x++) {
		for (int y = 0; y < img_h; y++) {
			cv::Point p(x, y);
			if (mask.at<uchar>(p)) {
				k1.at<float>(p) = fxx.at<float>(p) / std::pow(1 + std::pow(fx.at<float>(p), 2), 3 / 2.);
				k2.at<float>(p) = fyy.at<float>(p) / std::pow(1 + std::pow(fy.at<float>(p), 2), 3 / 2.);
				k3.at<float>(p) = f11.at<float>(p) / std::pow(1 + std::pow(f1.at<float>(p), 2), 3 / 2.);
				k4.at<float>(p) = f22.at<float>(p) / std::pow(1 + std::pow(f2.at<float>(p), 2), 3 / 2.);
			}
		}
	}

	// Scores
	int Wr = 0;
	cv::Mat Vt = cv::Mat::zeros(src.size(), CV_32FC1);
	int pos_end = 0;

	// Horizontal direction
	for (int y = 0; y < img_h; y++) {
		for (int x = 0; x < img_w; x++) {
			cv::Point p(x, y);
			bool bla = k1.at<float>(p) > 0;

			if (bla) {
				Wr++;
			}

			if (Wr > 0 && (x == img_w - 1 || !bla)) {
				if (x == img_w - 1) {
					pos_end = x;
				}
				else {
					pos_end = x - 1;
				}


				int pos_start = pos_end - Wr + 1;                                                                                                                                                                                         // Start pos of concave
				int pos_max = 0;
				float max = -FLT_MIN;
				for (int i = pos_start; i <= pos_end; i++ ) {
					float value = k1.at<float>(cv::Point(i, y));
					if (value > max) {
						pos_max = i;
						max = value;
					}
				}

				float Scr = k1.at<float>(cv::Point(pos_max, y)) * Wr;
				Vt.at<float>(cv::Point(pos_max, y)) +=  Scr;
				Wr = 0;
			}
		}
	}

	// Vertical direction
	for (int x = 0; x < img_w; x++) {
		for (int y = 0; y < img_h; y++) {
			cv::Point p(x, y);
			bool bla = k2.at<float>(p) > 0;

			if (bla) {
				Wr++;
			}

			if (Wr > 0 && (y == img_h - 1 || !bla)) {
				if (y == img_h - 1) {                                                                                                                                                                                 // TODO in ORIGINAL, there is X
					pos_end = y;
				}
				else {
					pos_end = y - 1;
				}


				int pos_start = pos_end - Wr + 1;                                                                                                                                                                                         // Start pos of concave

				int pos_max = 0;
				float max = FLT_MIN;
				for (int i = pos_start; i <= pos_end; i++ ) {
					float value = k2.at<float>(cv::Point(x, i));
					if (value > max) {
						pos_max = i;
						max = value;
					}
				}

				float Scr = k2.at<float>(cv::Point(x, pos_max)) * Wr;
				Vt.at<float>(cv::Point(x, pos_max)) +=  Scr;
				Wr = 0;
			}
		}
	}

	int pos_x_end = 0;
	int pos_y_end = 0;

	// Direction \ .
	for (int start = 0; start < img_h + img_w - 1; start++) {
		// Initial values
		int x, y;
		if (start < img_w) {
			x = start;
			y = 0;
		}
		else {
			x = 0;
			y = start - img_w + 1;
		}

		bool done = false;

		while (!done) {
			cv::Point p(x, y);
			bool bla = k3.at<float>(p) > 0;
			if (bla) {
				Wr++;
			}

			if (Wr > 0 && (y == img_h - 1 || x == img_w - 1 || !bla)) {
				if (y == img_h - 1 || x == img_w - 1) {
					// Reached edge of image
					pos_x_end = x;
					pos_y_end = y;
				}
				else {
					pos_x_end = x - 1;
					pos_y_end = y - 1;
				}

				int pos_x_start = pos_x_end - Wr + 1;
				int pos_y_start = pos_y_end - Wr + 1;


//				std::cout << pos_x_start << "\t" << pos_y_start<< "\t" << pos_x_end << "\t" << pos_y_end << "\n";
				cv::Rect rect(pos_x_start, pos_y_start, pos_x_end - pos_x_start + 1, pos_y_end - pos_y_start + 1);
				cv::Mat dd = k3(rect);
				cv::Mat d = dd.diag(0);

				float max = FLT_MIN;
				int pos_max = 0;
				for (int i = 0; i < d.rows; i++ ) {
					float value = d.at<float>(cv::Point(0, i));
					if (value > max) {
						pos_max = i;
						max = value;
					}
				}

				int pos_x_max = pos_x_start + pos_max;
				int pos_y_max = pos_y_start + pos_max;
				float Scr = k3.at<float>(cv::Point(pos_x_max, pos_y_max)) * Wr;

				Vt.at<float>(cv::Point(pos_x_max, pos_y_max)) +=  Scr;
				Wr = 0;
			}

			if ((x == img_w - 1) || (y == img_h - 1)) {
				done = true;
			}
			else {
				x++;
				y++;
			}
		}
	}


	// Direction /
	for (int start = 0; start < img_h + img_w - 1; start++) {
		// Initial values
		int x, y;
		if (start < img_w) {
			x = start;
			y = img_h - 1;
		}
		else {
			x = 0;
			y = img_w + img_h - start - 2;
		}

		bool done = false;

		while (!done) {

			cv::Point p(x, y);
			bool bla = k4.at<float>(p) > 0;
			if (bla) {
				Wr++;
			}

			if (Wr > 0 && (y == 0 || x == img_w - 1 || !bla)) {
				if (y == 0 || x == img_w - 1) {
					// Reached edge of image
					pos_x_end = x;
					pos_y_end = y;
				}
				else {
					pos_x_end = x - 1;
					pos_y_end = y + 1;
				}

				int pos_x_start = pos_x_end - Wr + 1;
				int pos_y_start = pos_y_end + Wr - 1;


				cv::Rect rect(pos_x_start, pos_y_end, pos_x_end - pos_x_start + 1, pos_y_start - pos_y_end + 1);

				// BUG: http://opencv-users.1802565.n2.nabble.com/Get-a-subimage-from-cv-Mat-td6656231.html
				cv::Mat roi = cv::Mat(k4, rect);
				cv::Mat dd;
				cv::flip(roi, dd, 0);
				cv::Mat d = dd.diag(0);

//				_printMatrix<float>(d);

				float max = FLT_MIN;
				int pos_max = 0;
				for (int i = 0; i < d.rows; i++ ) {
					float value = d.at<float>(cv::Point(0, i));
					if (value > max) {
						pos_max = i;
						max = value;
					}
				}

				int pos_x_max = pos_x_start + pos_max;
				int pos_y_max = pos_y_start - pos_max;


//				std::cout << pos_max << "\t";
//				std::cout << pos_x_max << "\t";
//				std::cout << pos_y_max << std::endl;

				float Scr = k4.at<float>(cv::Point(pos_x_max, pos_y_max)) * Wr;

				if (pos_y_max < 0) pos_y_max = 0;

				Vt.at<float>(cv::Point(pos_x_max, pos_y_max)) +=  Scr;
				Wr = 0;

			}

			if ((x == img_w - 1) || (y == 0)) {
				done = true;
			}
			else {
				x++;
				y--;
			}
		}
	}


	cv::Mat Cd1 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat Cd2 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat Cd3 = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat Cd4 = cv::Mat::zeros(src.size(), CV_32F);
	for (int x = 2; x < src.cols - 3; x++) {
		for (int y = 2; y < src.rows - 3; y++) {
			cv::Point p(x, y);
			Cd1.at<float>(p) = std::min(
					std::max(Vt.at<float>(cv::Point(x + 1, y)), Vt.at<float>(cv::Point(x + 2, y))),
					std::max(Vt.at<float>(cv::Point(x - 1, y)), Vt.at<float>(cv::Point(x - 2, y)))
					);
			Cd2.at<float>(p) = std::min(
					std::max(Vt.at<float>(cv::Point(x, y + 1)), Vt.at<float>(cv::Point(x, y + 2))),
					std::max(Vt.at<float>(cv::Point(x, y - 1)), Vt.at<float>(cv::Point(x, y - 2)))
					);
			Cd3.at<float>(p) = std::min(
					std::max(Vt.at<float>(cv::Point(x - 1, y - 1)), Vt.at<float>(cv::Point(x - 2, y - 2))),
					std::max(Vt.at<float>(cv::Point(x + 1, y + 1)), Vt.at<float>(cv::Point(x + 2, y + 2)))
					);
			Cd4.at<float>(p) = std::min(
					std::max(Vt.at<float>(cv::Point(x - 1, y + 1)), Vt.at<float>(cv::Point(x - 2, y + 2))),
					std::max(Vt.at<float>(cv::Point(x + 1, y - 1)), Vt.at<float>(cv::Point(x + 2, y - 2)))
					);
		}
	}

	// Connection of vein centres

	cv::Mat veins = cv::Mat::zeros(src.size(), CV_32F);
	for (int x = 0; x < src.cols; x++) {
		for (int y = 0; y < src.rows - 3; y++) {
			cv::Point p(x, y);
			veins.at<float>(p) =
			std::max(
			std::max(Cd1.at<float>(p), Cd2.at<float>(p)),
			std::max(Cd3.at<float>(p), Cd4.at<float>(p))
						);
		}
	}

	_dst.create(veins.size(), CV_32F);
	cv::Mat dst = _dst.getMat();
	veins.copyTo(dst);




//	_printMatrix<float>(veins);

//	std::vector<float> greaterThanZeroValues;
//	for (int i = 0; i < veins.size().width; i++) {
//		for (int j = 0; j < veins.size().height; j++) {
//			if (veins.at<float>(cv::Point(i,j)) > 0) {
//				greaterThanZeroValues.push_back(veins.at<float>(cv::Point(i,j)));
//			}
//		}
//	}

//	std::sort(greaterThanZeroValues.begin(), greaterThanZeroValues.end());
//	float median = greaterThanZeroValues[greaterThanZeroValues.size()/4*3];


//	std::cout << median << std::endl;

//	_dst.create(src.size(), CV_8U);
//	cv::Mat dst = _dst.getMat();
//	for (int i = 0; i < veins.size().width; i++) {
//		for (int j = 0; j < veins.size().height; j++) {
//			uchar value = (veins.at<float>(cv::Point(i,j)) > median) ? 255 : 0;
//			dst.at<uchar>(cv::Point(i,j)) = value;
//		}
//	}
}
