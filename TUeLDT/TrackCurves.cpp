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


void TrackingLaneDAG_generic::trackCurves(cv::Mat& map, int withFiltering)
{
#ifdef DEBUG_FILTERS
    if (debugX == 0) imshow("map", map);
#endif // DEBUG_FILTERS

    cv::Mat highPass;
	GaussianBlur(map, highPass, cv::Size(5, 5), 1, 1);
	addWeighted(map, 10.5, highPass, -11, 0, highPass);
	//multiply(map, highPass, map, 1/255.0);
#ifdef DEBUG_FILTERS
    if (debugX == 0) imshow("highPass", highPass);
#endif // DEBUG_FILTERS

	cv::Point r1, r2, l1, l2;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + map.rows;

	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;
	r2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + map.rows;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + map.rows;

	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;
	l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - 720 + map.rows;

	vector<Point2f> laneR, laneL, laneRxy, laneLxy;

	cv::Point c1(-3, 0);
	cv::Point c2(3, 0);
	cv::Point c3(0, 3);
	cv::Point c4(0, -3);


	BirdView bird;
	Mat input;
	Mat birdRaw;
	Mat filtered;

	bird.configureTransform(l1, l2, r1, r2, 600, BIRD_WIDTH, BIRD_HEIGHT);

	map.copyTo(input);

	birdRaw = bird.applyTransformation(input);

	if (withFiltering)
	{
		Mat birdHighPass;
		Mat birdWithoutCars;
		Mat tmp, tmp2;
		GaussianBlur(birdRaw, birdHighPass, cv::Size(7, 7), 1, 10);
		addWeighted(birdRaw, 11, birdHighPass, -11, 0, birdHighPass);
		GaussianBlur(birdHighPass, birdHighPass, cv::Size(3, 3), 0.5, 0.5);
		GaussianBlur(birdHighPass, tmp, cv::Size(3,3), 1, 1);

		tmp.copyTo(tmp2);
		translateImg(tmp2,-3, 0);
		translateImg(tmp,3, 0);

		absdiff(tmp, tmp2, tmp);

		tmp *= 4;
		tmp = 255 - tmp;

		multiply(tmp, birdHighPass, birdWithoutCars, 1.0/255);
		multiply(birdRaw, birdWithoutCars, birdWithoutCars, 1.0/255*2.0);

		GaussianBlur(birdWithoutCars, birdWithoutCars, cv::Size(3,3), 0.5, 0.5);
		birdWithoutCars.copyTo(filtered);

#ifdef DEBUG_FILTERS
if (debugX == 0) imshow("birdHighPass", birdHighPass);
if (debugX == 0) imshow("birdWithoutCars", birdWithoutCars);
#endif // DEBUG_FILTERS

	}
	else
	{
		map.copyTo(filtered);
	}

	// border - to avoid seg fault when calculating score
	cv::Mat mask = cv::Mat::zeros(filtered.rows, filtered.cols, CV_8U);
	mask(Rect(2, 2, filtered.cols - 4, filtered.rows - 4)) = 255;
	bitwise_and(filtered, mask, filtered);

#ifdef DEBUG_FILTERS
	if (debugX == 0) imshow("birdRaw", birdRaw);
#endif // DEBUG_FILTERS

	CurveDetector3 leftDet, rightDet;
	leftDet.name = "Left";
	rightDet.name = "Right";
	try
	{
		lsd.run(filtered);
	}
	catch(const char* msg)
	{
		cerr << "LSD exception: " << msg << "\n";
	}

	rightDet.seg = &lsd.seg;
	leftDet.seg = &lsd.seg;


	Point p(200 * BIRD_SCALE - 1, BIRD_HEIGHT - 1);
	rightDet.detectCurve(birdRaw, p, laneR);

	p = Point(150 * BIRD_SCALE - 1 , BIRD_HEIGHT - 1);
	leftDet.detectCurve(birdRaw, p, laneL);

	bird.invertPoints(laneR, mPtrLaneModel->curveR);
	bird.invertPoints(laneL, mPtrLaneModel->curveL);

	///////// DEBUG WINDOW //////////////////////////////////////

	if (debugX == 0)
	{
		cv::Mat FrameDbg;
		cv::cvtColor(map, FrameDbg, cv::COLOR_GRAY2BGR);

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
		mPtrLaneModel->curveR[i].y += this->mCAMERA.RES_VH[0] - map.rows;
	}

	for (size_t i = 0; i < mPtrLaneModel->curveL.size(); i++)
	{
		mPtrLaneModel->curveL[i].y += this->mCAMERA.RES_VH[0] - map.rows;
	}
}
