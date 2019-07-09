/*
 * MapGenerator.h
 *
 *  Created on: Jun 13, 2019
 *      Author: msz
 */

#ifndef TUELDT_MAPGENERATOR_H_
#define TUELDT_MAPGENERATOR_H_

#include "opencv2/opencv.hpp"


struct ProbMap{
	cv::Mat val;
	cv::Mat tan;
	cv::Mat nn;
};


class MapGenerator {
public:
	MapGenerator();
	void run(cv::Mat input, struct ProbMap * out);
	virtual ~MapGenerator();

	int gaussKernelSize;
	int gaussSigma;

	int grayTippingPoint; // if zero then it's not taken into account
	int magTippingPoint;



	cv::Mat mGradX;
	cv::Mat mGradY;
	cv::Mat mMask;
	cv::Mat mGradX_abs;
	cv::Mat mGradY_abs;
	cv::Mat mFrameGradMag;
	cv::Mat mTmp;
	cv::Mat mTempProbMat;
	cv::Mat mProbMap_Gray;
	cv::Mat mProbMap_GradMag;
};



#endif /* TUELDT_MAPGENERATOR_H_ */
