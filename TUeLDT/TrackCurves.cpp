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

#define DEBUG_FILTERS

// https://stackoverflow.com/questions/19068085/shift-image-content-with-opencv
Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return img;
}

const float BIRD_SCALE = 1;
const int BIRD_WIDTH = 350 * BIRD_SCALE;
const int BIRD_HEIGHT = 700 * BIRD_SCALE;


CustomLineSegmentDetector lsd(BIRD_WIDTH, BIRD_HEIGHT);


void TrackingLaneDAG_generic::trackCurves(cv::Mat& input, int withFiltering)
{
#ifdef DEBUG_FILTERS
    if (debugX == 0) imshow("trackCurves - input", input);
#endif // DEBUG_FILTERS

	cv::Point r1, r2, l1, l2;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + input.rows;

	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;
	r2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + input.rows;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + input.rows;

	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;
	l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + input.rows;

	vector<Point2f> laneR, laneL, laneRxy, laneLxy;

	cv::Point c1(-3, 0);
	cv::Point c2(3, 0);
	cv::Point c3(0, 3);
	cv::Point c4(0, -3);

	BirdView bird;
	Mat birdRaw;
	Mat blurred;


	vector<Point2f> startPoints, startPointsBird;

	bird.configureTransform(Point2f(180, 519), Point2f(180+210, 319), Point2f(1200, 519), Point2f(1200-210, 319), 600, BIRD_WIDTH, BIRD_HEIGHT);

	birdRaw = bird.applyTransformation(input);
	GaussianBlur(birdRaw, blurred, cv::Size(3,3), 1, 1);

	// border - to avoid seg fault when calculating score
	cv::Mat mask = cv::Mat::zeros(blurred.rows, blurred.cols, CV_8U);
	mask(Rect(2, 2, blurred.cols - 4, blurred.rows - 4)) = 255;
	bitwise_and(blurred, mask, blurred);

#ifdef DEBUG_FILTERS
	if (debugX == 0) imshow("birdRaw", birdRaw);
#endif // DEBUG_FILTERS

	CurveDetector3 leftDet, rightDet;
	leftDet.name = "Left";
	rightDet.name = "Right";
	try
	{
		lsd.run(blurred);
	}
	catch(const char* msg)
	{
		cerr << "LSD exception: " << msg << "\n";
	}

	rightDet.seg = &lsd.seg;
	leftDet.seg = &lsd.seg;


	startPoints.push_back(r1);
	startPoints.push_back(r2);
	startPoints.push_back(l1);
	startPoints.push_back(l2);

	bird.convertPointsToBird(startPoints, startPointsBird);



	rightDet.detectCurve2(blurred, startPointsBird[0], startPointsBird[1], laneR);
	leftDet.detectCurve2(blurred, startPointsBird[2], startPointsBird[3], laneL);

	bird.invertPoints(laneR, mPtrLaneModel->curveR);
	bird.invertPoints(laneL, mPtrLaneModel->curveL);

	///////// DEBUG WINDOW //////////////////////////////////////

	if (debugX == 0)
	{
		cv::Mat FrameDbg;
		cv::cvtColor(input, FrameDbg, cv::COLOR_GRAY2BGR);

		for(int i = 1; i < (int)mPtrLaneModel->curveR.size(); i++)
		{
			line(FrameDbg, mPtrLaneModel->curveR[i-1], mPtrLaneModel->curveR[i], CvScalar(255, 0, 0), 2);
		}

		for(int i = 1; i < (int)mPtrLaneModel->curveL.size(); i++)
		{
			line(FrameDbg, mPtrLaneModel->curveL[i-1], mPtrLaneModel->curveL[i], CvScalar(255, 0, 0), 2);
		}

		for (Point pt : mPtrLaneModel->curveR)
		{
			line(FrameDbg, pt + c1, pt + c2, CvScalar(0, 0, 200), 2);
			line(FrameDbg, pt + c3, pt + c4, CvScalar(0, 0, 200), 2);
		}

		for (Point pt : mPtrLaneModel->curveL)
		{
			line(FrameDbg, pt + c1, pt + c2, CvScalar(0, 0, 200), 2);
			line(FrameDbg, pt + c3, pt + c4, CvScalar(0, 0, 200), 2);
		}

		imshow("debug", FrameDbg);
	}

	for (size_t i = 0; i < mPtrLaneModel->curveR.size(); i++)
	{
		mPtrLaneModel->curveR[i].y += this->mCAMERA.RES_VH[0] - input.rows;
	}

	for (size_t i = 0; i < mPtrLaneModel->curveL.size(); i++)
	{
		mPtrLaneModel->curveL[i].y += this->mCAMERA.RES_VH[0] - input.rows;
	}
}
