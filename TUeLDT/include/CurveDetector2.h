/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#ifndef TUELDT_INCLUDE_CURVEDETECTOR2_H_
#define TUELDT_INCLUDE_CURVEDETECTOR2_H_

#include "opencv2/opencv.hpp"

using namespace cv;

class CurveDetector2
{



private:
	inline int isPointOutOfRange(Point a, int width, int height);
	int MAX_STEPS_AMOUNT;


public:
	int detectCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &curve);
	void grabPoints(Point a, Point b, std::vector<Point> &points);
	std::vector<Point> selectNextPoints(const cv::UMat& img, Point a, Point2f vec, int step);
	int calcScore(const cv::UMat& img, Point a, Point b, float d);
	int computeCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &curve);
	int left;
	std::vector< std::vector<cv::Point>> debugCurves;


};




#endif /* TUELDT_INCLUDE_CURVEDETECTOR2_H_ */
