/*
 * TrackCurves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#include "TrackingLaneDAG_generic.h"
#include "CurveDetector2.h"

extern int debugX, debugY, debugZ;

void TrackingLaneDAG_generic::trackCurves(cv::UMat& FrameRGB)
{
	cv::Mat mask;
	cv::Mat filtered;
	cv::Mat out;

	cv::Mat hsv;
	cv::Mat tmp;
	cv::Mat colors;

    cvtColor(FrameRGB, hsv, COLOR_BGR2HSV);
	//const size_t CH_HUE = 0;
    const size_t CH_SATURATION = 1;
    const size_t CH_VALUE = 2;
    const size_t BUFFER_SIZE = 5;

	// cv::Mat yellow_lines;
    // inRange(hsv, Scalar(10, 70, 40), Scalar(40, 255, 255), yellow_lines);

    Mat channels[3];
    split(hsv, channels);

    // REMOVE ASPHALT COLOR
    {
		const int MARGIN = 50;
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		int maxS, maxV;
		Mat image_roi;

		cv::Point r1, r2, l1, l2;

		r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
		r1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;

		r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;
		r2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;

		l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
		l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;

		l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;
		l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;

		cv::Point ravg = Point((r1+r2)/2);
		cv::Point lavg = Point((l1+l2)/2);

		int rx, ry, rw, rh;

		rx = lavg.x + MARGIN;
		ry = lavg.y;
		rw = ravg.x - lavg.x - 2* MARGIN;
		rh = l1.y - lavg.y;

		Rect region_of_interest = Rect(rx, ry, rw, rh);

		{ // DEBUG
			vector<cv::Point> area;

			area.push_back(Point(rx, ry+rh));
			area.push_back(Point(rx, ry));
			area.push_back(Point(rx+rw, ry));
			area.push_back(Point(rx+rw, ry+rh));

			mPtrLaneModel->debugCurves.clear();
			mPtrLaneModel->debugCurves.push_back(area);
		}

		image_roi = channels[CH_SATURATION](region_of_interest);
		minMaxLoc( image_roi, &minVal, &maxVal, &minLoc, &maxLoc );
		maxS = maxVal;

		image_roi = channels[CH_VALUE](region_of_interest);
		minMaxLoc( image_roi, &minVal, &maxVal, &minLoc, &maxLoc );
		maxV = maxVal;

		inRange(hsv, Scalar(0, 0, 0), Scalar(180, maxS, maxV), colors);
		colors = 255 - colors;

		bitwise_and(colors, channels[CH_VALUE], channels[CH_VALUE]);
    }

	imshow("out", channels[CH_VALUE]);

	if (debugX == 0) imshow("channels[CH_VALUE]", channels[CH_VALUE]);


	cv::Rect lROI;
	lROI = cv::Rect(0, mCAMERA.RES_VH(0) - mSPAN, mCAMERA.RES_VH(1), mSPAN);
	channels[CH_VALUE](lROI).copyTo(mFrameGRAY_ROI);
	mFrameGRAY_ROI /= BUFFER_SIZE;
	if (!buffer[0].cols)
	{
		for (size_t i = 0; i < BUFFER_SIZE; i++)
			mFrameGRAY_ROI.copyTo(buffer[i]);
		bufferIt = 0;
	}
	mFrameGRAY_ROI.copyTo(tmp);

	for (size_t i = 0; i < BUFFER_SIZE; i++)
		add(buffer[i], tmp, tmp);
	mFrameGRAY_ROI.copyTo(buffer[bufferIt]);

	bufferIt = (bufferIt + 1) % BUFFER_SIZE;

	cv::UMat utmp;
	tmp.copyTo(utmp);

	CurveDetector2 lcd, rcd;

	cv::Point r1, r2, l1, l2;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - FrameRGB.rows + tmp.rows;

	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;
	r2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - FrameRGB.rows + tmp.rows;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - FrameRGB.rows + tmp.rows;

	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;
	l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - FrameRGB.rows + tmp.rows;

	lcd.left = 1;
	rcd.left = -1;

	mPtrLaneModel->curveRight.clear();
	mPtrLaneModel->curveLeft.clear();
	GaussianBlur( utmp, utmp, cv::Size( 5, 5 ), 2, 2, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED  );

	rcd.detectCurve(utmp, r1, r2, mPtrLaneModel->curveRight);
	lcd.detectCurve(utmp, l1, l2, mPtrLaneModel->curveLeft);

	if ((mPtrLaneModel->curveRight.size() > 2) && (mPtrLaneModel->curveLeft.size() > 2))
	{
		Point2f l1 = mPtrLaneModel->curveLeft[0];
		Point2f l2 = mPtrLaneModel->curveLeft[1];

		Point2f r1 = mPtrLaneModel->curveRight[0];
		Point2f r2 = mPtrLaneModel->curveRight[1];

		Point2f crossPoint = rcd.findCrossPoint(l1, l2, r1, r2);

		if (crossPoint.x == 0) return;

		Point2f vl = crossPoint - l1;
		Point2f vr = crossPoint - r1;

		float dl_len = sqrt(vl.x*vl.x + vl.y*vl.y);
		float dr_len = sqrt(vr.x*vr.x + vr.y*vr.y);

		float len = dl_len;
		if (dr_len < len) len = dr_len;

		vl /= dl_len;
		vr /= dr_len;

		vl *= -len;
		vr *= -len;

		vl += crossPoint;
		vr += crossPoint;

		//line(FrameDbg, Point(vl), Point(vr), CvScalar(0, 0, 200), 2);

		Point2f cc = (vr + vl) / 2.0;

		Point2f tmpV = crossPoint - cc;
		tmpV *= 0.95;

		Point2f dd = cc + tmpV;

		Point2f kl = rcd.findCrossPoint(l1, l2, dd, dd + (vr - vl));
		Point2f kr = rcd.findCrossPoint(r1, r2, dd, dd + (vr - vl));

		Point2f el = kl - (kr - kl);
		Point2f er = kr + (kr - kl);
		//line(FrameDbg, Point(el), Point(er), CvScalar(0, 0, 200), 2);
		Point2f bl = vl - (vr - vl);
		Point2f br = vr + (vr - vl);

	    // Input Quadilateral or Image plane coordinates
	    Point2f inputQuad[4];
	    // Output Quadilateral or World plane coordinates
	    Point2f outputQuad[4];

	    // Lambda Matrix
	    Mat lambda( 2, 4, CV_32FC1 );
	    //Input and Output Image;
	    Mat input, output;

	    channels[CH_VALUE](lROI).copyTo(input);
	   // tmp.copyTo(input);

	    // Set the lambda matrix the same type and size as input
	    lambda = Mat::zeros( input.rows, input.cols, input.type() );

	    // Note that points in inputQuad and outputQuad have to be from top-left in clockwise order
	    inputQuad[0] = el;
	    inputQuad[1] = er;
	    inputQuad[2] = br;
	    inputQuad[3] = bl;

	    outputQuad[0] = Point2f( 0,0 );
	    outputQuad[1] = Point2f( input.cols-1,0);
	    outputQuad[2] = Point2f( input.cols-1,input.rows-1);
	    outputQuad[3] = Point2f( 0,input.rows-1  );

	    lambda = getPerspectiveTransform( inputQuad, outputQuad );
	    warpPerspective(input,output,lambda,output.size() );

	    if (debugX == 0) imshow("Output", output);
	}


	if (debugX == 0)
	{
		cv::Mat FrameDbg;
		cv::cvtColor(tmp, FrameDbg, cv::COLOR_GRAY2BGR);

		cv::Point c1(-3, 0);
		cv::Point c2(3, 0);
		cv::Point c3(0, 3);
		cv::Point c4(0, -3);

		for (vector<cv::Point> v : rcd.debugCurves)
		{
			for (size_t i = 1; i < v.size(); i++)
			{
				line(FrameDbg, v[i-1], v[i],  CvScalar(128, 0, 0), 2);
			}
		}

		for (vector<cv::Point> v : lcd.debugCurves)
		{
			for (size_t i = 1; i < v.size(); i++)
			{
				line(FrameDbg, v[i-1], v[i],  CvScalar(128, 0, 0), 2);
			}
		}

		for(int i = 1; i < (int)mPtrLaneModel->curveRight.size(); i++)
		{
			line(FrameDbg, mPtrLaneModel->curveRight[i-1], mPtrLaneModel->curveRight[i], CvScalar(255, 0, 0), 2);
		}

		for(int i = 1; i < (int)mPtrLaneModel->curveLeft.size(); i++)
		{
			line(FrameDbg, mPtrLaneModel->curveLeft[i-1], mPtrLaneModel->curveLeft[i], CvScalar(255, 0, 0), 2);
		}

		for (Point pt : mPtrLaneModel->curveRight)
		{
			line(FrameDbg, pt + c1, pt + c2, CvScalar(0, 0, 200), 2);
			line(FrameDbg, pt + c3, pt + c4, CvScalar(0, 0, 200), 2);
		}

		for (Point pt : mPtrLaneModel->curveLeft)
		{
			line(FrameDbg, pt + c1, pt + c2, CvScalar(0, 0, 200), 2);
			line(FrameDbg, pt + c3, pt + c4, CvScalar(0, 0, 200), 2);
		}

		imshow("debug", FrameDbg);
	}

	for (size_t i = 0; i < mPtrLaneModel->curveRight.size(); i++)
	{
		mPtrLaneModel->curveRight[i].y += FrameRGB.rows - tmp.rows;
	}

	for (size_t i = 0; i < mPtrLaneModel->curveLeft.size(); i++)
	{
		mPtrLaneModel->curveLeft[i].y += FrameRGB.rows - tmp.rows;
	}
}
