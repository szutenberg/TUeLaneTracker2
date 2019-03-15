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
	quant(2),
	angTh(22.5),
	logEps(0),
	densityTh(0.7),
	nBins(1024)
{
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
		int NFA = res[i*7+6];
		if (p1.y < p2.y) swap(p1, p2);
		cv::Point2f shift;
		int score = calcScore(p1, p2, shift);
		p1 += shift;
		p2 += shift;
		if (score > maxScore) maxScore = score;

		cv::Point2f vec = p2 - p1;
		float angle = atan(vec.x / vec.y) * 180.0 / 3.14;

		tmp.a = p1;
		tmp.b = p2;
		tmp.angle = angle;
		tmp.score = score;
		tmp.NFA = NFA;
		seg.push_back(tmp);
	}

	maxScore /= 100.0;
	for (int i = 0; i < *n_out; i++)
	{
		seg[i].score /= maxScore;
	}

	sort(seg.begin(), seg.end(), compareScores);

	delete n_out;
	delete [] res;

	return true;
}

const int SHIFT_FROM = -2;
const int SHIFT_TO = 2;
const int SHIFT_N = SHIFT_TO - SHIFT_FROM + 1;

int CustomLineSegmentDetector::calcScore(cv::Point2f a, cv::Point2f b, cv::Point2f& shift)
{
	if (a == b) return 0;
	int dirx = 0;
	int diry = 0;
	int bins[SHIFT_N];
	for (int i = 0; i < SHIFT_N; i++) bins[i] = 0;
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

	if (from < 0) from = 0;
	if ((dirx) && (to >= mWidth)) to = mWidth - 1;
	if ((diry) && (to >= mHeight)) to = mHeight - 1;

	float p = diry * a.x + dirx * a.y;

	for (int i = from; i <= to; i++, p+=slope, counter++)
	{
		for(int bin = SHIFT_FROM; bin <= SHIFT_TO; bin++)
		{
			int j = 0.5 + p + bin;
			int px = dirx * i + diry * j;
			int py = dirx * j + diry * i;
			bins[bin - SHIFT_FROM] += mImgI[px + py * mWidth];
		}
	}

	int maxVal = -1;
	int maxBin;

	for (int i = 0; i < SHIFT_N; i++)
	{
		if (maxVal < bins[i])
		{
			maxBin = i;
			maxVal = bins[i];
		}
	}

	shift.x = diry * (maxBin + SHIFT_FROM);
	shift.y = dirx * (maxBin + SHIFT_FROM);
	return bins[maxBin] / counter;
}


CustomLineSegmentDetector::~CustomLineSegmentDetector() {
	// TODO Auto-generated destructor stub
	delete [] mImgD;
	delete [] mImgI;
}

