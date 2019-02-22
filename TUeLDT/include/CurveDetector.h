/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#ifndef TUELDT_INCLUDE_CURVEDETECTOR_H_
#define TUELDT_INCLUDE_CURVEDETECTOR_H_

#include "opencv2/opencv.hpp"

using namespace cv;

class CurveDetector
{



private:




public:
	int detectCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &curve);
	std::vector<Point> grabPoints(Point a, Point b);
	std::vector<Point> selectNextPoints(const cv::UMat& img, Point a, Point2f vec);
	int calcScore(const cv::UMat& img, Point a, Point b);






};




#endif /* TUELDT_INCLUDE_CURVEDETECTOR_H_ */
