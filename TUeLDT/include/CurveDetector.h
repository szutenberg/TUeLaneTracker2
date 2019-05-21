/*
 * CurveDetector.h
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#ifndef TUELDT_CURVEDETECTOR_H_
#define TUELDT_CURVEDETECTOR_H_

#include "Config.h"
#include "LaneModel.h"
#include "Templates.h"
#include "BirdView.h"

class CurveDetector {


public:
	CurveDetector(const LaneTracker::Config* cfg, const LaneFilter* laneFilter, const VanishingPtFilter* vpFilter, const Templates* templates);
	virtual ~CurveDetector();
	int run(cv::UMat& frame, LaneModel* Lane);

private:
	void prepareROI(cv::UMat input, cv::UMat& output);
	void blur(cv::UMat input, cv::UMat& output);
	void computeMap(cv::Mat& input, cv::Mat& outputMag, cv::Mat& outputAng);
	void setParams(LaneModel* Lane, Mat roi);
	const LaneTracker::Config* mCfg;
	const LaneFilter* mLaneFilter;
	const VanishingPtFilter* mVpFilter;
	const Templates* mTemplates;
	double pxPerCm;
	void evaluateC(double *dest, int N);

	float lastConfidence;
	double lastAl, lastBl, lastCl;
	double lastAr, lastBr, lastCr;

	cv::UMat mInput;
	cv::UMat mROI;
	cv::UMat mROIblurred;
	cv::Mat mGradX;
	cv::Mat mGradY;
	cv::Mat mMask;
	cv::Mat mGradX_abs;
	cv::Mat mGradY_abs;
	cv::Mat mFrameGradMag;

	cv::Mat mFrMag, mFrAng;
	cv::Mat mTempProbMat;
	cv::Mat mAngMap;
	cv::Mat mProbMap_Gray;
	cv::Mat mProbMap_GradMag;
	cv::Mat mProbVal;
	cv::Mat mProbMap_GradDir;

	cv::Point2f mVp;
	cv::Point2f mLeft, mRight;
	cv::Point2f baseL, baseR, defaultVp;
	cv::Point2f r1, r2, l1, l2;
	cv::Point2f birdLeftStart, birdRightStart;

	cv::Mat bufMag[10];
	cv::Mat bufAng[10];
	int bufIt;
	BirdView mBird;

	float leftB, rightB;
	vector<cv::Point2f> segmentsToRemove;

	cv::Mat tmp, tmp2;
	void findCandidates(double *in, double *params, int* amt);
	double calcHistProb(double *in, double a, double b);
	void plotArray(const char * name, double *in, int N);
	void filterHistorgrams(double* in, double *out);
	void plotHistograms(const char * name, double* vals);
	void calculateHistograms(cv::Mat img, cv::Mat ang, double* vals);
	vector<int> curve(double a, double b, double c);
	void calculateMatch(double *dest, int N, double a, double b, Mat img, Mat ang);

	void trackEgoLane(double *c, int N);
	void filterLaneMarkings(cv::Mat img, cv::Mat& laneMarkingsVal);
    void matchParabolaWithMap(cv::Mat mFrMag, cv::Mat mFrAng, double maxA, double maxB, double values[], int N);
    void applyConfidenceGradient(cv::Mat& img);
    void bufferFiltering(int bufferSize);
	static int TIPPING_POINT_GRAY;
	static int TIPPING_POINT_GRAD_Mag;
	int mROIy; /* y-coordinate of upper edge of ROI in the input image */

};

#endif /* TUELDT_CURVEDETECTOR_H_ */
