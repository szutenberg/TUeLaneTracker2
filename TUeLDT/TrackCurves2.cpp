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



const int BUF_SIZE = 8;
int bufIt = 0;
cv::Mat buf[BUF_SIZE];


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


void TrackingLaneDAG_generic::trackCurves2(cv::Mat& input)
{
#ifdef DEBUG_BIRD
    if (debugX == 0) imshow("trackCurves2 - input", input);
#endif // DEBUG_BIRD

	cv::Point r1, r2, l1, l2;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;
	r2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;
	l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	BirdView bird;
	Mat birdRaw;
	Mat filtered;

	bird.configureTransform(Point2f(180, 519), Point2f(180+210, 319), Point2f(1200, 519), Point2f(1200-210, 319), 600, BIRD_WIDTH, BIRD_HEIGHT);
	birdRaw = bird.applyTransformation(input);

	if (debugX == 0) imshow("input", birdRaw);
	GaussianBlur(birdRaw, birdRaw, cv::Size(5,5), 1, 1);
	if (debugX == 0) imshow("inputBlur", birdRaw);

	Sobel( birdRaw, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( birdRaw, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

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
	subtract(birdRaw, mLaneMembership.TIPPING_POINT_GRAY, mTempProbMat, cv::noArray(), CV_32S);
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
	//	subtract(mGradTanTemplate, mBufferPool->GradientTangent[mBufferPos], mTempProbMat, cv::noArray(), CV_32S);
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


	Mat prob;
	mBufferPool->Probability[mBufferPos].copyTo(prob);

	if (buf[0].rows == 0)
	{
		for (int i = 0; i < BUF_SIZE; i++) prob.copyTo(buf[i]);
		bufIt = 0;
	}

	prob.copyTo(buf[bufIt]);
	bufIt = (bufIt+1)%BUF_SIZE;


	prob.copyTo(filtered);
	filtered *= 0;

	for (int i = 0; i < BUF_SIZE; i++)
		addWeighted(filtered, 1, buf[i], 1.0/BUF_SIZE, 1, filtered);

	GaussianBlur(filtered, filtered, cv::Size(15,1), 0, 0);


	Mat filteredDbg;

	filtered.copyTo(filteredDbg);
    cvtColor(filteredDbg, filteredDbg, COLOR_GRAY2BGR);


	vector<Point2f> startPoints, startPointsBird;
	startPoints.push_back(r1);
	startPoints.push_back(r2);
	startPoints.push_back(l1);
	startPoints.push_back(l2);

	bird.convertPointsToBird(startPoints, startPointsBird);

	vector<Point2f> cL, cR;
	Point2f v;
	cL.push_back(startPointsBird[2]);
	cR.push_back(startPointsBird[0]);

	cout << startPointsBird;


	int y = cL[0].y - 200;
	int dL = 0;
	int dR = 0;
	for(int i = 0; i < 5; i++)
	{
		Point lL = cL[cL.size()-1];
		Point lR = cR[cR.size()-1];

		float maxS = 0;

		float tL[101];
		float tR[101];
		float angleCoeff;

		for (int j = -50; j <= 50; j++)
		{
			Point leftSide(-10, 0);
			Point rightSide(10, 0);
			float l = calcScore(filtered, lL, Point2f(lL.x + j, y)) * 4.0;
			//l -= calcScore(filtered, lL+leftSide, Point(lL.x + j, y)+leftSide);
			//l -= calcScore(filtered, lL+rightSide, Point(lL.x + j, y)+rightSide);

			float r = calcScore(filtered, lR, Point2f(lR.x + j, y)) * 4.0;
			//r -= calcScore(filtered, lR+leftSide, Point(lR.x + j, y)+leftSide);
			//r -= calcScore(filtered, lR+rightSide, Point(lR.x + j, y)+rightSide);

			angleCoeff = 1.0 - abs(dL - j) / (abs(dL - j) + 10.0);
			tL[j+50] = l * angleCoeff;

			angleCoeff = 1.0 - abs(dR - j) / (abs(dR - j) + 10.0);
			tR[j+50] = r * angleCoeff;
			//cout << j << "\t" << l << "\t" << r << endl;
		}

		maxS = 0;
		for (int a = -50; a <= 50; a++)
		{
			for (int b = -50; b <= 50; b++)
			{
				float change = abs(a-b);
				float score = tL[a+50] * tR[b+50] * (1.0 - change / (change + 10.0));

				if (score > maxS)
				{
					maxS = score;
					dL = a;
					dR = b;
				}
			}
		}
		//cout << i << ":" << maxS << endl;
		cL.push_back(Point2f(lL.x + dL, y));
		cR.push_back(Point2f(lR.x + dR, y));
		y -= 200;
	}


#ifdef DEBUG_BIRD
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
