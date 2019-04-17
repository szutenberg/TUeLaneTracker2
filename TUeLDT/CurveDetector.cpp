/*
 * CurveDetector.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#include "CurveDetector.h"
#include "opencv2/opencv.hpp"

using namespace cv;

CurveDetector::CurveDetector(const LaneTracker::Config* cfg) : mCfg(cfg){
	// TODO Auto-generated constructor stub

}

CurveDetector::~CurveDetector() {
	// TODO Auto-generated destructor stub
}

int CurveDetector::run(cv::UMat& frame, LaneModel* Lane)
{
	imshow("curveDet", frame);

	Lane->curveL.clear();
	Lane->curveR.clear();

	Lane->curveL.push_back(Point2f(480, 480));
	Lane->curveL.push_back(Point2f(0, 0));

	Lane->curveR.push_back(Point2f(480+100, 480));
	Lane->curveR.push_back(Point2f(0+100, 0));

	return 0;
}
