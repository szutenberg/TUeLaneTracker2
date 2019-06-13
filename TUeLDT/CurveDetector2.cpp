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

using namespace cv;

CurveDetector2::CurveDetector2(const LaneTracker::Config* cfg,
		const LaneFilter* laneFilter,
		const VanishingPtFilter* vpFilter,
		const Templates* templates) :
		mCfg(cfg),
		mLaneFilter(laneFilter),
		mVpFilter(vpFilter),
		mTemplates(templates){
}

CurveDetector2::~CurveDetector2() {
	// TODO Auto-generated destructor stub
}


vector<Point2f> histogramOld(Mat mag, Mat ang, Point2f point)
{
	vector<Point2f> ret;
	imshow("histogram", mag);
	Mat tmp, mask;
	float maxV = 0.000001;
	for (int i = -200; i < 200; i+=2)
	{
		int tanMin = (float)(i-5) * 128.0 / mag.rows;
		int tanMax = (float)(i+5) * 128.0 / mag.rows;
		mag.copyTo(tmp);
		mask = ang < tanMin;
		tmp.setTo(0, mask);
		mask = ang > tanMax;
		tmp.setTo(0, mask);

		double val = cv::sum( tmp )[0];
		if (val > maxV) maxV = val;
		ret.push_back(Point2f(i, val));
	}

	for (int i = 0; i < ret.size(); i++) ret[i].y /= maxV;

	return ret;
}


vector<Point2f> histogram(Mat mag, Point2f point, float shift)
{
	vector<Point2f> ret;


	Mat show;
	mag.copyTo(show);
	line(show, point, Point2f(point.x, 0), cvScalar(130), 3);



	Mat tmp, mask, lt;
	mag.convertTo(mag, CV_16U);
	mag.copyTo(lt);

	int yH = mag.rows / 10;
	for (int i = 0; i < 11; i++)
	{
		line(lt, Point(0, mag.rows-i*yH+yH/2), Point(mag.cols-1, mag.rows-i*yH+yH/2), cvScalar(i*25), yH/2+2);
	}

	multiply(mag, lt, mag);

	//imshow("histogram", mag);


	mag.copyTo(lt);

	for (int i = -200; i < 200; i+=5)
	{
		lt *= 0;
		Point2f lc = Point2f(i, point.y);
		lc /= 2;

		line(lt, point, Point2f(point.x + i + shift, 0), cvScalar(1), 15);

		multiply(lt, mag, lt);
		double val = cv::sum( lt )[0];

		val *= exp(-i*i/5000);

		if (val < 0) val = 0;
		ret.push_back(Point2f(i, val));
	}

	return ret;
}



int CurveDetector2::run(TrackingLaneDAG_generic& tr, LaneModel* Lane)
{
	imshow("input frame", tr.mProbMapFocussed);




	imshow("input frame ang", tr.mGradTanFocussed);

	int difY = mCfg->cam_res_v - tr.mProbMapFocussed.rows;

	const int MARGIN_WIDTH = 300;
	cv::Mat FrameTest;
	FrameTest = cv::Mat::zeros(tr.mProbMapFocussed.rows, 2*MARGIN_WIDTH + mCfg->cam_res_h, CV_8U);
	tr.mProbMapFocussed.copyTo(FrameTest(cv::Rect(MARGIN_WIDTH, 0, tr.mProbMapFocussed.cols, tr.mProbMapFocussed.rows)));

	Mat workspace;
	FrameTest.copyTo(workspace);

	cv::cvtColor(FrameTest, FrameTest, cv::COLOR_GRAY2BGR);

	uint32_t mHistPurviewMax = 0;
	uint32_t mHistBaseMax = 0;
	for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
	{
		if (tr.mHistPurview.at<uint32_t>(i) > mHistPurviewMax) mHistPurviewMax = tr.mHistPurview.at<uint32_t>(i);
		if (tr.mHistBase.at<uint32_t>(i)    > mHistBaseMax)    mHistBaseMax    = tr.mHistBase.at<uint32_t>(i);
	}

	uint32_t BAR_MAX_HEIGHT = 60;
	uint32_t mHistPurviewScale = mHistPurviewMax / BAR_MAX_HEIGHT;
	uint32_t mHistBaseScale = mHistBaseMax / BAR_MAX_HEIGHT;

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

	Point2f lBase(Lane->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x, mLaneFilter->BASE_LINE_ICCS    + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f rBase(Lane->boundaryRight[0]+ mLaneFilter->O_ICCS_ICS.x, mLaneFilter->BASE_LINE_ICCS    + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f lPur (Lane->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x, mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY);
	Point2f rPur (Lane->boundaryRight[1]+ mLaneFilter->O_ICCS_ICS.x, mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - difY);



	cv::line(FrameTest, lBase + Point2f(MARGIN_WIDTH, 0), lPur + Point2f(MARGIN_WIDTH, 0), cvScalar(255, 255, 0), 5);
	cv::line(FrameTest, rBase + Point2f(MARGIN_WIDTH, 0), rPur + Point2f(MARGIN_WIDTH, 0), cvScalar(255, 255, 0), 5);


	const int HIST_HEIGHT = 70;
	const int HIST_WIDTH = 200;

	Point2f lDif = lBase - lPur;
	Point2f rDif = rBase - rPur;

	lDif /= lDif.y / HIST_HEIGHT;
	rDif /= rDif.y / HIST_HEIGHT;
	lDif.x *= -1;
	rDif.x *= -1;

	Mat mag;



	Rect sel(lPur.x - lDif.x+MARGIN_WIDTH-HIST_WIDTH/2, lPur.y - HIST_HEIGHT, HIST_WIDTH, HIST_HEIGHT);
	workspace(sel).copyTo(mag);

	vector<Point2f> hL = histogram(mag, Point2f(HIST_WIDTH/2+lDif.x, HIST_HEIGHT-1), lDif.x);

	sel = Rect(rPur.x - rDif.x+MARGIN_WIDTH-HIST_WIDTH/2, rPur.y - HIST_HEIGHT, HIST_WIDTH, HIST_HEIGHT);
	workspace(sel).copyTo(mag);

	vector<Point2f> hR = histogram(mag, Point2f(HIST_WIDTH/2+rDif.x, HIST_HEIGHT-1), rDif.x);

	float maxV = 0.000001;
	for (Point2f pt : hL)
		if (pt.y > maxV) maxV = pt.y;
	for (Point2f pt : hR)
		if (pt.y > maxV) maxV = pt.y;


	const int BAR_H = 40;
	for (Point2f pt : hL)
	{
		cv::line(FrameTest, lPur + Point2f(MARGIN_WIDTH + pt.x + lDif.x, -HIST_HEIGHT), lPur + Point2f(MARGIN_WIDTH + pt.x + lDif.x, -pt.y/maxV * BAR_H-HIST_HEIGHT), cvScalar(255, 255, 0), 2);
	}

	for (Point2f pt : hR)
	{
		cv::line(FrameTest, rPur + Point2f(MARGIN_WIDTH + pt.x + rDif.x, -HIST_HEIGHT), rPur + Point2f(MARGIN_WIDTH + pt.x + rDif.x, -pt.y/maxV * BAR_H-HIST_HEIGHT), cvScalar(0, 255, 255), 2);
	}

	cerr << lDif << " " << rDif << endl;








	cv::imshow("hist", FrameTest);




/*
	test  = curve(lastAl, lastBl, lastCl);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}
	Lane->curveL.clear();

	vector<Point2f> out;
	test  = curve(lastAl, lastBl, lastCl);
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}

	mBird.invertPoints(out, Lane->curveL);
	out.clear();

	test  = curve(lastAr, lastBr, lastCr);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}

	test  = curve(lastAr, lastBr, lastCr);
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}
	mBird.invertPoints(out, Lane->curveR);

	for (Point2f & pt : Lane->curveL) pt += Point2f(0, mROIy);
	for (Point2f & pt : Lane->curveR) pt += Point2f(0, mROIy);

	if (mCfg->display_graphics)  imshow("debugFrame", debugFrame);

*/

	return 0;
}
