/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#ifndef TUELDT_INCLUDE_CURVEDETECTOR3_H_
#define TUELDT_INCLUDE_CURVEDETECTOR3_H_

#include "opencv2/opencv.hpp"
#include "CustomLineSegmentDetector.h"
using namespace cv;

class CurveDetector3
{

public:
	int detectCurve(const cv::Mat& img, Point start, std::vector<Point2f> &lane);
	std::string name;
	std::vector<LineSegment>* seg;

	std::vector< std::vector<cv::Point>> debugCurves;
};


#endif /* TUELDT_INCLUDE_CURVEDETECTOR3_H_ */
