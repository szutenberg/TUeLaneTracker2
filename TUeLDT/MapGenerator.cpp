/*
 * MapGenerator.cpp
 *
 *  Created on: Jun 13, 2019
 *      Author: msz
 */

#include "MapGenerator.h"
#include "NeuralNetwork.h"
using namespace cv;

MapGenerator::MapGenerator() {
	// TODO Auto-generated constructor stub

	gaussKernelSize = 5;
	gaussSigma = 2;
	grayTippingPoint = 100;
	magTippingPoint = 40;

}

MapGenerator::~MapGenerator() {
	// TODO Auto-generated destructor stub
}


void MapGenerator::run(Mat input, struct ProbMap* out)
{
	GaussianBlur( input, input, cv::Size( gaussKernelSize, gaussKernelSize ), gaussSigma, gaussSigma, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED  );

	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

	mMask = mGradX> 255;
	mGradX.setTo(255, mMask);
	mMask = mGradX <-255;
	mGradX.setTo(-255, mMask);
	mMask = mGradX ==0;
	mGradX.setTo(1, mMask);

	mMask = mGradY> 255;
	mGradY.setTo(255, mMask);
	mMask = mGradY <-255;
	mGradY.setTo(-255, mMask);
	mMask = mGradY ==0;
	mGradY.setTo(1, mMask);

	//convert to absolute scale and add weighted absolute gradients
	mGradX_abs = abs(mGradX);
	mGradY_abs = abs(mGradY);

	mFrameGradMag = mGradX_abs + mGradY_abs;

	//convertScaleAbs(mFrameGradMag, mFrameGradMag);
	mFrameGradMag.convertTo(mFrameGradMag, CV_8U);

	cv::divide(mGradX, mGradY, out->tan, 128, -1);

	//GrayChannel Probabilities
	subtract(input, grayTippingPoint, mTempProbMat, cv::noArray(), CV_32S);
	mMask = mTempProbMat <0 ;
	mTempProbMat.setTo(0,mMask);
	mTempProbMat.copyTo(mProbMap_Gray);
	mTempProbMat = mTempProbMat + 10;

	divide(mProbMap_Gray, mTempProbMat, mProbMap_Gray, 255, -1);


	//GradientMag Probabilities
	subtract(mFrameGradMag, magTippingPoint, mTempProbMat, cv::noArray(), CV_32S);
	mTempProbMat.copyTo(mProbMap_GradMag);
	mTempProbMat= abs(mTempProbMat) + 10;
	divide(mProbMap_GradMag, mTempProbMat, mProbMap_GradMag, 255, -1);


	// Intermediate Probability Map
	mTmp  = mProbMap_GradMag + mProbMap_Gray;
	mMask = mTmp  <0 ;
	mTmp .setTo(0, mMask);

	mTmp.convertTo(out->val, CV_8U, 0.5);

}

