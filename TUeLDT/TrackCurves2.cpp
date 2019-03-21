/*
 * TrackCurves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#include "TrackingLaneDAG_generic.h"
#include "CurveDetector2.h"
#include "CurveDetector3.h"
#include "BirdView.h"

extern int debugX, debugY, debugZ;

#define DEBUG_BIRD

const float BIRD_SCALE = 1.4 ;
const int BIRD_WIDTH = 350 * BIRD_SCALE;
const int BIRD_HEIGHT = 700 * BIRD_SCALE;


float calcScore(cv::Mat img, cv::Point2f a, cv::Point2f b)
{
	if (a == b) return 0;
	int dirx = 0;
	int diry = 0;

	int counter = 0;
	float slope = (float)(b.y - a.y) / (b.x - a.x);

	if (abs(slope) < 1)
	{
		dirx = 1;
		if (a.x > b.x)
		{
			swap(a, b);
		}
	}
	else
	{
		diry = 1;
		if (a.y > b.y)
		{
			swap(a, b);
		}
		slope = 1 / slope;
	}

	int from = dirx * a.x + diry * a.y;
	int to   = dirx * b.x + diry * b.y;

	float p = diry * a.x + dirx * a.y;
	int ret = 0;

	for (int i = from; i <= to; i++, p+=slope, counter++)
	{
		int j = p;
		int px = dirx * i + diry * j;
		int py = dirx * j + diry * i;
		int val = (int)img.at<unsigned char>(py, px);
		val *= val;
		ret += val;
	}

	return (float)ret / counter;
}

void drawPointsX(cv::Mat& img, vector<Point2f> points)
{
	cv::Point c1(-3, 0);
	cv::Point c2(3, 0);
	cv::Point c3(0, 3);
	cv::Point c4(0, -3);

	for (Point pt : points)
	{
		line(img, pt + c1, pt + c2, CvScalar(0, 0, 200), 2);
		line(img, pt + c3, pt + c4, CvScalar(0, 0, 200), 2);
	}
}


cv::Mat TrackingLaneDAG_generic::createProbabilityMap(cv::Mat input)
{
	// TODO use different variables when making parallel
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
	mGradY_abs = abs(mGradY );

	//addWeighted( mGradX_abs, 0.5, mGradY_abs, 0.5, 0, mFrameGradMag );
	mFrameGradMag = mGradX_abs + mGradY_abs;

	//convertScaleAbs(mFrameGradMag, mFrameGradMag);
	mFrameGradMag.convertTo(mFrameGradMag, CV_8U);

	cv::divide(mGradY, mGradX, mBufferPool->GradientTangent[mBufferPos], 128, -1);

	//GrayChannel Probabilities
	subtract(input, mLaneMembership.TIPPING_POINT_GRAY, mTempProbMat, cv::noArray(), CV_32S);
	mMask = mTempProbMat <0 ;
	mTempProbMat.setTo(0,mMask);
	mTempProbMat.copyTo(mProbMap_Gray);
	mTempProbMat = mTempProbMat + 10;

	divide(mProbMap_Gray, mTempProbMat, mProbMap_Gray, 255, -1);

	//GradientMag Probabilities
	subtract(mFrameGradMag, mLaneMembership.TIPPING_POINT_GRAD_Mag, mTempProbMat, cv::noArray(), CV_32S);
	mTempProbMat.copyTo(mProbMap_GradMag);
	mTempProbMat= abs(mTempProbMat) + 10;
	divide(mProbMap_GradMag, mTempProbMat, mProbMap_GradMag, 255, -1);

	// Intermediate Probability Map
	mBufferPool->Probability[mBufferPos] = mProbMap_GradMag + mProbMap_Gray;
	mMask = mBufferPool->Probability[mBufferPos] <0 ;
	mBufferPool->Probability[mBufferPos].setTo(0,mMask);
	mBufferPool->Probability[mBufferPos].copyTo(mProbMapNoTangent);

	//Gradient Tangent Probability Map
	//	subtract(mGradTanTemplatescore, mBufferPool->GradientTangent[mBufferPos], mTempProbMat, cv::noArray(), CV_32S);
	// We have one value for whole map due to bird transformation
	// First we try with angle = 0;
	mBufferPool->GradientTangent[mBufferPos].convertTo(mTempProbMat, CV_32S);

	mTempProbMat= abs(mTempProbMat);
	mTempProbMat.copyTo(mProbMap_GradDir);
	mTempProbMat = mTempProbMat + 10;
	divide(mProbMap_GradDir, mTempProbMat, mProbMap_GradDir, 255, -1);
	subtract(255, mProbMap_GradDir, mProbMap_GradDir, cv::noArray(), -1);


	//Final Probability Map
	multiply(mBufferPool->Probability[mBufferPos], mProbMap_GradDir, mBufferPool->Probability[mBufferPos]);
	mBufferPool->Probability[mBufferPos].convertTo(mBufferPool->Probability[mBufferPos], CV_8U, 1.0/255, 0);

	return mBufferPool->Probability[mBufferPos];
}



void TrackingLaneDAG_generic::trackCurves2(cv::Mat& input)
{
	cv::Point r1, r2, l1, l2;
	BirdView bird;
	Mat birdRaw;
	Mat buffered;
	Mat prob; // probability map

#ifdef DEBUG_BIRD
    if (debugX == 0) imshow("trackCurves2 - input", input);
#endif // DEBUG_BIRD

	r1.y = l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;
	r2.y = l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;


	bird.configureTransform(Point2f(180, 519), Point2f(180+210, 319), Point2f(1200, 519), Point2f(1200-210, 319), 600, BIRD_WIDTH, BIRD_HEIGHT);
	birdRaw = bird.applyTransformation(input);

	if (debugX == 0) imshow("input", birdRaw);
	GaussianBlur(birdRaw, birdRaw, cv::Size(5,5), 1, 1);
	if (debugX == 0) imshow("inputBlur", birdRaw);

	prob = createProbabilityMap(birdRaw);

	if (mBuf[0].rows == 0)
	{
		for (int i = 0; i < mBUF_SIZE; i++) prob.copyTo(mBuf[i]);
		mBufIt = 0;
	}
	prob.copyTo(mBuf[mBufIt]);
	mBufIt = (mBufIt+1)%mBUF_SIZE;

	buffered = Mat::zeros(prob.rows, prob.cols, CV_8U);

	for (int i = 0; i < mBUF_SIZE; i++)
		addWeighted(buffered, 1, mBuf[i], 1.0/mBUF_SIZE, 1, buffered);

	GaussianBlur(buffered, buffered, cv::Size(15,5), 0, 0);

	vector<Point2f> startPoints, startPointsBird;
	startPoints.push_back(r1);
	startPoints.push_back(r2);
	startPoints.push_back(l1);
	startPoints.push_back(l2);

	bird.convertPointsToBird(startPoints, startPointsBird);

	vector<Point2f> cL, cR;
	cL.push_back(startPointsBird[2]);
	cR.push_back(startPointsBird[0]);


	if (debugY == 0) debugY = 180;

	int y = cL[0].y - debugY;
	int rangeX = 50;

	int dL = 0;
	int dR = 0;
	for(int i = 0; i < cL[0].y/debugY; i++)
	{
		Point lL = cL[cL.size()-1];
		Point lR = cR[cR.size()-1];

		float maxS = 0;
		float scoreL[2*rangeX + 1];
		float scoreR[2*rangeX + 1];

		for (int j = -rangeX; j <= rangeX; j++)
		{
			float l = 0;
			float r = 0;

			for (int shift = -3; shift <= 3; shift++)
			{
				l += calcScore(buffered, lL+Point(shift, 0), Point2f(lL.x + j + shift, y));
				r += calcScore(buffered, lR+Point(shift, 0), Point2f(lR.x + j + shift, y));
			}

			scoreL[j+rangeX] = l;
			scoreR[j+rangeX] = r;
		}

		maxS = 0;

		int newdL;
		int newdR;
		for (int iL = -rangeX; iL <= rangeX; iL++)
		{
			for (int iR = -rangeX; iR <= rangeX; iR++)
			{
				float change = abs(iL - iR);
				float score = scoreL[iL + rangeX] * scoreR[iR + rangeX];
				float lAngle = 1.0 - abs(dL - iL) / (abs(dL - iL) + 10.0);
				float rAngle = 1.0 - abs(dR - iR) / (abs(dR - iR) + 10.0);
				score *= lAngle * rAngle;
				score *= (1.0 - change / (change + 10.0));

				if (score > maxS)
				{
					maxS = score;
					newdL = iL;
					newdR = iR;
				}

			}
		}

		dL = newdL;
		dR = newdR;

		cL.push_back(Point2f(lL.x + dL, y));
		cR.push_back(Point2f(lR.x + dR, y));
		y -= debugY;
	}


#ifdef DEBUG_BIRD
	Mat filteredDbg;
	buffered.copyTo(filteredDbg);
    cvtColor(filteredDbg, filteredDbg, COLOR_GRAY2BGR);

	drawPointsX(filteredDbg, cL);
	drawPointsX(filteredDbg, cR);
	if (debugX == 0) imshow("filteredDbg", filteredDbg);
#endif // DEBUG_BIRD

	bird.invertPoints(cR, mPtrLaneModel->curveR);
	bird.invertPoints(cL, mPtrLaneModel->curveL);

	for (size_t i = 0; i < mPtrLaneModel->curveR.size(); i++)
	{
		mPtrLaneModel->curveR[i].y += this->mCAMERA.RES_VH[0] - input.rows;
	}

	for (size_t i = 0; i < mPtrLaneModel->curveL.size(); i++)
	{
		mPtrLaneModel->curveL[i].y += this->mCAMERA.RES_VH[0] - input.rows;
	}
}
