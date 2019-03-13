/*
 * CustomLineSegmentDetector.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#include "CustomLineSegmentDetector.h"

#include <lsd_1.6/lsd.h>
#include <lsd_1.6/lsd.c> // fix me

CustomLineSegmentDetector::CustomLineSegmentDetector(int width, int height):
	mHeight(height),
	mWidth(width),
	mSize(width * height),
	mImgI(NULL),
	mImgD(NULL),
	quant(5),
	angTh(22.5),
	logEps(0),
	densityTh(0.5),
	nBins(1024)
{
	// TODO Auto-generated constructor stub

	mImgI = new int[mSize];
	mImgD = new double[mSize];
}


bool CustomLineSegmentDetector::run(cv::Mat img)
{
	if (!img.isContinuous())
		throw "Image is not continuous!";

	if ((img.cols != mWidth) || (img.rows != mHeight))
		throw "Image has different size than expected!";

	uchar * ptr = img.data;
	for (size_t i = 0; i < mSize; i++)
	{
		mImgD[i] = ptr[i];
		mImgI[i] = ptr[i];
	}

	int * n_out = new int;
	double * res = LineSegmentDetection(
					n_out, mImgD, mWidth, mHeight,
					1, 1,  // no resizing
					quant, angTh, logEps, densityTh, nBins,
					NULL, NULL, NULL // we don't want to get pixel mapping
				);

	double maxScore = 0;

	seg.clear();

	for (int i = 0; i < *n_out; i++)
	{
		LineSegment tmp;
		cv::Point2f p1(res[i*7+0], res[i*7+1]);
		cv::Point2f p2(res[i*7+2], res[i*7+3]);
		if (p1.y < p2.y) swap(p1, p2);
		int score = calcScore(img, cv::Point(p1+cv::Point2f(0.5, 0.5)), cv::Point(p2+cv::Point2f(0.5, 0.5)));
		if (score > maxScore) maxScore = score;

		cv::Point2f vec = p2 - p1;
		float angle = atan(vec.x / vec.y) * 180.0 / 3.14;

		tmp.a = p1;
		tmp.b = p2;
		tmp.angle = angle;
		tmp.score = score;
		seg.push_back(tmp);
	}

	maxScore /= 100.0;
	for (int i = 0; i < *n_out; i++)
	{
		seg[i].score /= maxScore;
	}

	delete n_out;
	delete [] res;

	return true;
}


int CustomLineSegmentDetector::calcScore(const cv::Mat& img, cv::Point a, cv::Point b)
{
	int maxVal = 0;
	int res = 0;
	if (a == b) return 0;
	float slope = (float)(b.y - a.y) / (b.x - a.x);
	int dirx = 0;
	int diry = 0;
	int swapped = 0;
	if (abs(slope) < 1)
	{
		dirx = 1;
		if (a.x > b.x)
		{
			swap(a, b);
			swapped = 1;
		}
	}
	else
	{
		diry = 1;
		if (a.y > b.y)
		{
			swap(a, b);
			swapped = 1;
		}
		slope = 1 / slope;
	}

	int from = dirx * a.x + diry * a.y;
	int to   = dirx * b.x + diry * b.y;

	if (from < 0) from = 0;
	if ((dirx) && (to >= img.cols)) to = img.cols - 1;
	if ((diry) && (to >= img.rows)) to = img.rows - 1;

	int counter = 0;

	float p = diry * a.x + dirx * a.y;

	int sum = 0;

	for (int i = from; i <= to; i++, p+=slope)
	{
		for (int j = p-2; j <= p+2; j++, counter++)
		{
			cv::Point pos(dirx * i + diry * j, dirx * j + diry * i);
			res += img.at<uchar>(pos);
		}
	}

	return res / counter;
}


CustomLineSegmentDetector::~CustomLineSegmentDetector() {
	// TODO Auto-generated destructor stub
	delete [] mImgD;
	delete [] mImgI;
}

