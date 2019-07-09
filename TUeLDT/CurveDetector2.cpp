/*
 * CurveDetector2.cpp
 *
 *  Created on: Jun 11, 2019
 *      Author: Michal Szutenberg
 */

#include "CurveDetector2.h"
#include "opencv2/opencv.hpp"
#include "BirdView.h"
#include "NeuralNetwork.h"
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
using namespace cv;


CurveDetector2::CurveDetector2(const LaneTracker::Config* cfg,
		const LaneFilter* laneFilter,
		const VanishingPtFilter* vpFilter,
		const Templates* templates) :
		mCfg(cfg),
		mLaneFilter(laneFilter),
		mVpFilter(vpFilter),
		mTemplates(templates){
	bufIt = 0;
	bufSize = 3;

	for (int i = 0 ; i < bufSize; i++)
	{
		buf[i].nn = Mat::zeros(	mTemplates->SPAN, mCfg->cam_res_h, CV_8U);
		buf[i].tan = Mat::zeros(mTemplates->SPAN, mCfg->cam_res_h, CV_16S);
		buf[i].val = Mat::zeros(mTemplates->SPAN, mCfg->cam_res_h, CV_8U);
	}

	ROI = cv::Rect(0, mCfg->cam_res_v - mTemplates->SPAN, mCfg->cam_res_h, mTemplates->SPAN);

	timeline = cv::Mat::zeros(1000, mLaneFilter->COUNT_BINS*2, CV_8U);
	timeline2 = cv::Mat::zeros(1000, mLaneFilter->COUNT_BINS*2, CV_8U);

	timelineIt = 0;
}

CurveDetector2::~CurveDetector2() {
	// TODO Auto-generated destructor stub
}

inline double det(double a, double b, double c, double d)
{
	return a*d - b*c;
}

Point2f findCrossPoint(Point2f a1, Point2f a2, Point2f b1, Point2f b2)
{
	Point2f ret(0,0);

	double detL1 = det(a1.x, a1.y, a2.x, a2.y);
	double detL2 = det(b1.x, b1.y, b2.x, b2.y);

	double xnom = det(detL1, a1.x - a2.x, detL2, b1.x - b2.x);
	double ynom = det(detL1, a1.y - a2.y, detL2, b1.y - b2.y);
	double denom = det( a1.x - a2.x, a1.y - a2.y, b1.x - b2.x, b1.y - b2.y);

	if(abs(denom) > 0.0001)
	{
		ret.x = xnom / denom;
		ret.y = ynom / denom;
	}

	return ret;
}

Mat imgROI;
Mat nnOutput;

int CurveDetector2::run(TrackingLaneDAG_generic& tr, LaneModel* Lane, Mat input)
{

	input(ROI).copyTo(imgROI);
	int difY = mCfg->cam_res_v - tr.mProbMapFocussed.rows;

	Point2f lBase(Lane->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x, mLaneFilter->BASE_LINE_ICCS    + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f rBase(Lane->boundaryRight[0]+ mLaneFilter->O_ICCS_ICS.x, mLaneFilter->BASE_LINE_ICCS    + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f lPur (Lane->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x, mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f rPur (Lane->boundaryRight[1]+ mLaneFilter->O_ICCS_ICS.x, mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f vp(Lane->vanishingPt.H + mLaneFilter->O_ICCS_ICS.x, Lane->vanishingPt.V + mLaneFilter->O_ICCS_ICS.y - difY);
	mapGen.run(imgROI, &buf[bufIt]);

	nnOutput = NeuralNetwork::getResult();
	if (nnOutput.cols)
	{
		nnOutput(cv::Rect(0, nnOutput.rows - buf[bufIt].val.rows, buf[bufIt].val.cols, buf[bufIt].val.rows)).copyTo(buf[bufIt].nn);
		multiply(buf[bufIt].nn, buf[bufIt].val, buf[bufIt].val, 1, CV_8U);
		if (mCfg->display_graphics) imshow("res", buf[bufIt].val);
	}


	bufIt++;
	bufIt %= bufSize;

	buf[0].val.copyTo(finalMap.val);
	buf[0].tan.copyTo(finalMap.tan);


	for (int i = 1; i < bufSize; i++)
	{
		mMask = buf[i].val > finalMap.val;

		buf[i].val.copyTo(finalMap.val, mMask);
		buf[i].tan.copyTo(finalMap.tan, mMask);
	}





	if (mCfg->display_graphics)
	{
		imshow("finalMap.val", finalMap.val);
		imshow("input", input);
		imshow("input frame", tr.mProbMapFocussed);

		const int MARGIN_WIDTH = 300;
		cv::Mat FrameTest;
		FrameTest = cv::Mat::zeros(tr.mProbMapFocussed.rows, 2*MARGIN_WIDTH + mCfg->cam_res_h, CV_8U);
		tr.mProbMapFocussed.copyTo(FrameTest(cv::Rect(MARGIN_WIDTH, 0, tr.mProbMapFocussed.cols, tr.mProbMapFocussed.rows)));


		Mat workspace;
		FrameTest.copyTo(workspace);


		cv::cvtColor(FrameTest, FrameTest, cv::COLOR_GRAY2BGR);

		float maxV = 0;
		for (int v = 0; v < mVpFilter->BINS_V.size(); v++)
		{
			for (int h = 0; h < mVpFilter->BINS_H.size(); h++)
			{
				if (tr.mTransitVpFilter.at<int32_t>(v ,h) > maxV) maxV = tr.mTransitVpFilter.at<int32_t>(v ,h);
			}
		}

		for (int v = 0; v < mVpFilter->BINS_V.size(); v++)
		{
			for (int h = 0; h < mVpFilter->BINS_H.size(); h++)
			{
				int val = tr.mTransitVpFilter.at<int32_t>(v ,h) / maxV * 255;
				Point pos = cvPoint(MARGIN_WIDTH + 320 + mVpFilter->BINS_H(h), mVpFilter->BINS_V(v)+ mLaneFilter->O_ICCS_ICS.y-difY);
				cv::line(FrameTest, pos, pos, cvScalar(val, 0, val), 4);
			}
		}


		uint32_t mHistPurviewMax = 0;
		uint32_t mHistBaseMax = 0;
		uint32_t mHistFarMax = 0;
		for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
		{
			if (tr.mHistPurview.at<uint32_t>(i) > mHistPurviewMax) mHistPurviewMax = tr.mHistPurview.at<uint32_t>(i);
			if (tr.mHistBase.at<uint32_t>(i)    > mHistBaseMax)    mHistBaseMax    = tr.mHistBase.at<uint32_t>(i);
			if (tr.mHistFar.at<uint32_t>(i)    > mHistFarMax)    mHistFarMax    = tr.mHistFar.at<uint32_t>(i);
		}

		uint32_t BAR_MAX_HEIGHT = 60;
		uint32_t mHistPurviewScale = mHistPurviewMax / BAR_MAX_HEIGHT;
		uint32_t mHistBaseScale = mHistBaseMax / BAR_MAX_HEIGHT;
		uint32_t mHistFarScale = mHistFarMax / BAR_MAX_HEIGHT;

		for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
		{
			int x = mLaneFilter->PURVIEW_BINS.at<int32_t>(i, 0) + mLaneFilter->O_ICCS_ICS.x;
			int y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY;
			int val = tr.mHistPurview.at<uint32_t>(i) / mHistPurviewScale;
			if (val < 0) val = 1;
			cv::line(FrameTest, cvPoint(x+MARGIN_WIDTH, y), cvPoint(x+MARGIN_WIDTH, y - val), cvScalar(0, 150, 0), 2);
		}

		for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
		{
			int x = mLaneFilter->BASE_BINS.at<int32_t>(i, 0) + mLaneFilter->O_ICCS_ICS.x;
			int y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY;
			int val = tr.mHistBase.at<uint32_t>(i) / mHistBaseScale;
			if (val < 0) val = 1;
			cv::line(FrameTest, cvPoint(x+MARGIN_WIDTH, y), cvPoint(x+MARGIN_WIDTH, y - val), cvScalar(150, 0, 0), 4);
		}

		for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
		{
			int x = mLaneFilter->FAR_BINS.at<int32_t>(i, 0) + mLaneFilter->O_ICCS_ICS.x;
			int y = mLaneFilter->FAR_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY;
			int val = tr.mHistFar.at<uint32_t>(i) / mHistFarScale;
			if (val < 0) val = 1;
			cv::line(FrameTest, cvPoint(x+MARGIN_WIDTH, y), cvPoint(x+MARGIN_WIDTH, y - val), cvScalar(150, 0, 0), 4);
		}
		line(FrameTest, vp+Point2f(MARGIN_WIDTH,-difY), vp+Point2f(MARGIN_WIDTH,-difY), cvScalar(0,255,0), 2);



		cv::line(FrameTest, lBase + Point2f(MARGIN_WIDTH, 0), lPur + Point2f(MARGIN_WIDTH, 0), cvScalar(255, 255, 0), 5);
		cv::line(FrameTest, rBase + Point2f(MARGIN_WIDTH, 0), rPur + Point2f(MARGIN_WIDTH, 0), cvScalar(255, 255, 0), 5);

		cv::imshow("hist", FrameTest);
	}



	int N = 20;

	Point2f ptsR[N][12];
	Point2f crPtR[N][7];
	float resultR[N];
	Point2f ptsL[N][12];
	Point2f crPtL[N][7];
	float resultL[N];

	if (mCfg->display_graphics)
	{
		Mat ransacDebug = ransac(rBase, rPur, 1, N, ptsR, crPtR, resultR);
		ransacDebug += ransac(lBase, lPur, -1, N, ptsL, crPtL, resultL);
		imshow("ranscac Debug", ransacDebug);
	}
	else
	{
		ransac(rBase, rPur, 1, N, ptsR, crPtR, resultR);
		ransac(lBase, lPur, -1, N, ptsL, crPtL, resultL);
	}


	float maxE = -1;
	int sR, sL;

	for (int l = 0; l < N; l++)
	{
		for (int r = 0; r < N; r++)
		{
			float score = resultR[r] * resultL[l];

			Point2f pR = ptsR[r][10];
			Point2f pL = ptsL[l][10];
			Point2f vR = ptsR[r][11];
			Point2f vL = ptsL[l][11];

			Point2f vpn = findCrossPoint(pR, pR + vR, pL, pL + vL);

			double arg = vpn.y - vp.y;
			double prob = exp(-arg*arg/(2.0 * 20.0 * 20.0));

			score *= prob;

			if (score > maxE)
			{
				maxE = score;
				sR = r;
				sL = l;

			}
		}
	}





	if (mCfg->display_graphics)
	{

	    Mat debugFr;
	    finalMap.val.copyTo(debugFr);
		cv::cvtColor(debugFr, debugFr, cv::COLOR_GRAY2BGR);


		for (int j = 1; j < 11; j++)
		{
			line(debugFr, ptsR[sR][j-1], ptsR[sR][j], cvScalar(255, 0,0), 2);
			line(debugFr, ptsL[sL][j-1], ptsL[sL][j], cvScalar(255, 0,0), 2);
		}


		Point2f pR = ptsR[sR][10];
		Point2f pL = ptsL[sL][10];
		Point2f vR = ptsR[sR][11];
		Point2f vL = ptsL[sL][11];

		Point2f vpn = findCrossPoint(pR, pR + vR, pL, pL + vL);

		line(debugFr, vpn, vpn, cvScalar(0, 255, 255), 2);
		line(debugFr, vp, vp, cvScalar(0, 255, 0), 2);

		double arg = vpn.y - vp.y;
		double prob = exp(-arg*arg/(2.0 * 20.0 * 20.0));
		//cerr << "prob = " << prob << endl;


		imshow("RANSAC", debugFr);
	}

	return 0;
}




Mat CurveDetector2::preparePotentialBoundaries(Point2f base, Point2f pur)
{
	Mat ret;
	Point2f v;
	v = pur - base;
	double corr = abs(2* (v.x / sqrt(v.x*v.x+v.y*v.y)));

	float H = 40 * corr;
	float R = 70;
	float H2 = 100 * corr;
	float R2 = 150;
	int polyN = 8;
	Point poly[8];

	poly[0] = (pur+base)*0.5 + Point2f(-15, 0);
	{
		v =  base - pur;
		v /= v.y;

		Point2f hP = Point(pur - v * H);
		v /= sqrt(v.x * v.x + v.y*v.y);

		float tmpVal;
		tmpVal = -v.y;
		v.y = v.x;
		v.x = tmpVal;

		poly[1] = Point(hP + v * R);
		poly[4] = Point(hP - v * R);
	}


	{
		v =  base - pur;
		v /= v.y;

		Point2f hP = Point(pur - v * H2);
		v /= sqrt(v.x * v.x + v.y*v.y);

		float tmpVal;
		tmpVal = -v.y;
		v.y = v.x;
		v.x = tmpVal;

		poly[2] = Point(hP + v * R2);
		poly[3] = Point(hP - v * R2);
	}
	poly[5] = (pur+base)*0.5 + Point2f(15, 0);
	poly[6] = base + Point2f(25, 0);
	poly[7] = base + Point2f(-25, 0);

	const Point* arrr;
	arrr = poly;
	const Point** arr;
	arr = &arrr;

	ret = cv::Mat::zeros(finalMap.val.rows, finalMap.val.cols, CV_8U);

	fillPoly(ret, arr, &polyN, 1, Scalar(255));
	return ret;
}



Mat CurveDetector2::ransac(Point2f base, Point2f pur, int rightSide, int N, Point2f pts[][12], Point2f crPt[][7], float result[])
{
	Mat linePlot;
	Mat triangle = preparePotentialBoundaries(base, pur);
	Mat copy;
	finalMap.val.copyTo(copy);
	copy += 1;
	multiply(triangle, copy, triangle, 1/255.0);

	Mat ind;
	ind = cv::Mat::zeros(finalMap.val.rows, finalMap.val.cols, CV_32S);

	int cnt = 0;
	int val;
	for (int y = 0; y < finalMap.val.rows; y++)
	{
		for (int x = 0; x < finalMap.val.cols; x++)
		{
			val = triangle.at<uint8_t>(y, x);
			cnt += val;
			ind.at<int32_t>(y, x) = cnt;
		}
	}

	linePlot = cv::Mat::zeros(finalMap.val.rows, finalMap.val.cols, CV_8U);


	float maxRes = 0;

    for (int i = 0; i < N; i++)
    {
    	Point2f P[4];

    	P[0] = base;
    	P[1] = pur;

    	for (int pt = 2; pt <= 3; pt++)
    	{
    		int minY, maxY, x, y;

    		if (pt == 2) minY = pur.y - 100, maxY = pur.y - 20;
    		if (pt == 3) minY = pur.y - 150, maxY = P[2].y - 20;

    		//minY = pur.y - 150;
    		//maxY = pur.y + 50;


    		int minVal = ind.at<int32_t>(minY, 0);
    		int maxVal = ind.at<int32_t>(maxY, ind.cols-1);

    		int range = maxVal - minVal;
    		if (range < 1) range = 1;
    		int res = rand() % range;
    		int index = res + minVal;

    		for (y = minY; y <= maxY; y++) // TODO binary search
    		{
    			if ((ind.at<int32_t>(y, ind.cols-1)) >= index) break;
    		}

    		for (x = 0; x < ind.cols; x++) // TODO binary search
    		{
    			if ((ind.at<int32_t>(y, x)) >= index) break;
    		}

    		P[pt] = Point(x, y);
    	}

    	if (P[2].y < P[3].y)
    	{
    		Point tmp  = P[2];
    		P[2] = P[3];
    		P[3] = tmp;
    	}
    	linePlot = 0;

		Point2f dt[11];
    	Point2f lp = P[0];

    	for (int it = 0; it <= 10; it++)
    	{
    		double t = 0.1 * (double)it;
    		Point2f np =  ((1 - t) * (1-t) * (1-t) * P[0] + 3.0 * (1-t) * (1-t) * t * P[1] + 3.0 * (1-t) * t * t * P[2] + t * t * t * P[3]);
    		pts[i][it] = np;

    		dt[it] = 3 * (1-t)*(1-t) * (P[1] - P[0]) + 6*(1-t)*t*(P[2]-P[1]) + 3 * t * t * (P[3] - P[2]);

    		if (it > 0)
    		{
    			line(linePlot, lp, np, cvScalar(255), 16);
    		}
    		lp = np;
    	}

    	pts[i][11] = dt[10];

    	//linePlot(Rect(0,0,linePlot.cols, 130)).setTo(0);
    	double amount = cv::sum( linePlot )[0];

    	if (amount < 1) amount = 1;

		multiply(linePlot, triangle, linePlot, 1/255.0);
		result[i] = cv::sum( linePlot )[0] / amount;

		imshow("linePlot", linePlot);

		if (maxRes < result[i]) maxRes = result[i];

		crPt[i][3] = P[0];
		crPt[i][4] = P[1];
		crPt[i][5] = P[2];
		crPt[i][6] = P[3];
    }

    for (int i = 0; i < N; i ++)
    {
    	result[i] /= maxRes;
    }

    return triangle;
}
