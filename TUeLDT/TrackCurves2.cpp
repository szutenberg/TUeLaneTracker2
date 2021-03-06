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


float histR[200];
float histL[200];
int histRange = 80;


void TrackingLaneDAG_generic::calcHistogram(cv::Point2f from, cv::Point2f to, cv::Mat pVal, cv::Mat pGrad, float* hist)
{
	int ymin = min(from.y, to.y);
	int ymax = max(from.y, to.y);
	int xmin = min(from.x, to.x);
	int xmax = max(from.x, to.x);

	for (int i = 0; i < 200; i++) hist[i] = 0;

	for (int j = ymin; j <= ymax; j++)
	{
		for (int i = xmin; i <= xmax; i++)
		{
			int value = (int)pVal.at<unsigned char>(j, i);
			int grad = (int)pGrad.at<short int>(j, i);


			if ((grad > 10) && (grad < 200)) hist[grad] += value;
			//hist[bin] += value;

		}
	}

	hist[histRange] += 0.00001;
}


float TrackingLaneDAG_generic::calcGradScore(cv::Point2f a, cv::Point2f b)
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

	int tab[256];
	for (int i = 0; i < 256; i++) tab[i] = 0;

	float myBin = (float)(a.x - b.x + b.y - a.y) / (b.y - a.y) * histRange;
	//cout << myBin << endl;
	float res = 0;

	for (int i = from; i <= to; i++, p+=slope, counter++)
	{
		int j = p;
		int px = dirx * i + diry * j;
		int py = dirx * j + diry * i;
		int val = (int)mProbVal.at<unsigned char>(py, px);
		int bin = (int)mProbBin.at<short int>(py, px);

		float coef = 1 - (float)abs(bin - myBin) / (abs(bin - myBin) + 10);

		//cout << bin << "\t" << val << "\t" << coef <<  endl;
		res += coef * val;
	}

	return res / counter;





}



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

	int tab[256];
	for (int i = 0; i < 256; i++) tab[i] = 0;


	for (int i = from; i <= to; i++, p+=slope, counter++)
	{
		int j = p;
		int px = dirx * i + diry * j;
		int py = dirx * j + diry * i;
		int val = (int)img.at<unsigned char>(py, px);
		ret += val;
		tab[val]++;
	}

	int sum = 0;
	int i = 0;
	while(sum * 2 < counter)
	{
		sum += tab[i++];
	}

	return i + 50;

	return (float)ret / counter + 60;
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


void TrackingLaneDAG_generic::createProbabilityMap(cv::Mat input)
{
	// TODO use different variables when making parallel
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

    mMask = mGradX ==0;
    mGradX.setTo(1, mMask);

	//convert to absolute scale and add weighted absolute gradients
	mGradX_abs = abs(mGradX);
	mGradY_abs = abs(mGradY);

	mFrameGradMag = mGradX_abs + mGradY_abs;
	mFrameGradMag.convertTo(mFrameGradMag, CV_8U);

	mGradY += mGradX; // we want to have (y+x)/x * histRange
	cv::divide(mGradY, mGradX, mProbBin, debugY, -1);
	mProbBin -= debugY - histRange;

	//mMask = mProbBin == histRange;
	//mProbBin.setTo(0, mMask);

	mMask = mProbBin < 0;
	mProbBin.setTo(0, mMask);

	mMask = mProbBin > (histRange*2);
	mProbBin.setTo(0, mMask);


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
	mProbVal = mProbMap_GradMag + mProbMap_Gray;
	mMask = mProbVal <0 ;
	mProbVal.setTo(0,mMask);

	//Gradient Tangent Probability Map
	//	subtract(mGradTanTemplatescore, mBufferPool->GradientTangent[mBufferPos], mTempProbMat, cv::noArray(), CV_32S);
	// We have one value for whole map due to bird transformation
	// First we try with angle = 0;
	mProbVal.convertTo(mTempProbMat, CV_32S);

	mTempProbMat= abs(mTempProbMat);
	mTempProbMat.copyTo(mProbMap_GradDir);
	mTempProbMat = mTempProbMat + 10;
	divide(mProbMap_GradDir, mTempProbMat, mProbMap_GradDir, 255, -1);
	subtract(255, mProbMap_GradDir, mProbMap_GradDir, cv::noArray(), -1);

	mProbMap_GradDir = Mat::ones(mTempProbMat.rows, mTempProbMat.cols, CV_32S) * 255;


	//Final Probability Map
	multiply(mProbVal, mProbMap_GradDir, mProbVal);
	mProbVal.convertTo(mProbVal, CV_8U, 1.0/255/2, 0);

}


void TrackingLaneDAG_generic::trackCurves2(cv::Mat& input)
{
	cv::Point r1, r2, l1, l2;
	BirdView bird;
	Mat birdRaw;


	if (debugY == 0) debugY = 170;


#ifdef DEBUG_BIRD
    if (debugX == 0) imshow("trackCurves2 - input", input);
#endif // DEBUG_BIRD

	r1.y = l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;
	r2.y = l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - this->mCAMERA.RES_VH[0] + input.rows;

	r1.x = mPtrLaneModel->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r2.x = mPtrLaneModel->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;

	l1.x = mPtrLaneModel->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l2.x = mPtrLaneModel->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;


    int binWidth = mLaneFilter->LANE.AVG_WIDTH/mLaneFilter->BINS_STEP_cm;
    int zeroPos = mLaneFilter->BASE_BINS.rows/2;
    int lPos = zeroPos - binWidth/2;
    int rPos = zeroPos + binWidth/2;

    assert(lPos >= 0);
    assert(rPos < mLaneFilter->BASE_BINS.rows);

    Point2f defaultVp = mLaneFilter->O_ICCS_ICS;
    defaultVp += Point2f(mLaneFilter->CAMERA.HORIZON_VH[1], mLaneFilter->CAMERA.HORIZON_VH[0]);
    defaultVp.y += - this->mCAMERA.RES_VH[0] + input.rows; // we have ROI

    Point2f baseL = defaultVp;
    baseL.y = l1.y;
    baseL.x += mLaneFilter->BASE_BINS.at<int>(0, lPos);

    Point2f baseR = defaultVp;
    baseR.y = r1.y;
    baseR.x += mLaneFilter->BASE_BINS.at<int>(0, rPos);

    float pixelsPerCm = (baseR.x - baseL.x) / mLaneFilter->LANE.AVG_WIDTH / mLaneFilter->BINS_STEP_cm;


#ifndef DEBUG_HORIZON
	Mat dbg;
	input.copyTo(dbg);
    cvtColor(dbg, dbg, COLOR_GRAY2BGR);

    vector<Point2f> pts;
    pts.push_back(baseL);
    pts.push_back(baseR);
    pts.push_back(defaultVp);
    drawPointsX(dbg, pts);
    //cerr << defaultVp << baseL << baseR << endl;
    if (debugX == 0) imshow("trackCurves2 - debug Horizon", dbg);
#endif // DEBUG_HORIZON

	bird.configureTransform2(baseL, baseR, defaultVp, BIRD_WIDTH, BIRD_HEIGHT);
	birdRaw = bird.applyTransformation(input);

	if (debugX == 0) imshow("input", birdRaw);
	GaussianBlur(birdRaw, birdRaw, cv::Size(5,5), 1, 1);
	if (debugX == 0) imshow("inputBlur", birdRaw);

	createProbabilityMap(birdRaw);

	vector<Point2f> startPoints, startPointsBird;
	startPoints.push_back(r1);
	startPoints.push_back(r2);
	startPoints.push_back(l1);
	startPoints.push_back(l2);
	startPoints.push_back(Point2f(input.cols, defaultVp.y));
	startPoints.push_back(Point2f(input.cols, input.rows));
	startPoints.push_back(Point2f(0, defaultVp.y));
	startPoints.push_back(Point2f(0, input.rows));
	bird.convertPointsToBird(startPoints, startPointsBird);

	line(mProbVal, startPointsBird[4],startPointsBird[5], CvScalar(0), 5);
	line(mProbVal, startPointsBird[6],startPointsBird[7], CvScalar(0), 5);
	line(mProbVal, startPointsBird[5],startPointsBird[7], CvScalar(0), 5);


	if (debugX == 0) imshow("prob", mProbVal);

	vector<Point2f> cL, cR;
	cR.push_back(startPointsBird[0]);
	Point2f avgP = (startPointsBird[0] + startPointsBird[1])/2;
	cR.push_back(avgP);

	cL.push_back(startPointsBird[2]);
	avgP = (startPointsBird[2] + startPointsBird[3])/2;
	cL.push_back(avgP);



	mBUF_SIZE = 3;

	if (mBufValue[0].rows == 0)
	{
		for (int i = 0; i < mBUF_SIZE; i++) mProbVal.copyTo(mBufValue[i]);
		for (int i = 0; i < mBUF_SIZE; i++) mProbBin.copyTo(mBufGrad[i]);
		mBufIt = 0;
	}
	mProbVal.copyTo(mBufValue[mBufIt]);
	mProbBin.copyTo(mBufGrad[mBufIt]);

	mBufIt = (mBufIt+1)%mBUF_SIZE;

	mBufValue[mBufIt].copyTo(mProbVal);
	mBufGrad[mBufIt].copyTo(mProbBin);

	for (size_t i = 1; i< (size_t)mBUF_SIZE; i++ )
	{
		int it = (i+mBufIt)%mBUF_SIZE;
		mMask = mProbVal < mBufValue[it];
		mBufValue[it].copyTo(mProbVal, mMask );
		mBufGrad[it].copyTo(mProbBin, mMask );
	}

	Mat filteredDbg;
	mProbVal.copyTo(filteredDbg);
    cvtColor(filteredDbg, filteredDbg, COLOR_GRAY2BGR);


	int y = cL[1].y - debugY;
	int rangeX = 60;

	int dL = (cL[1].x - cL[0].x)/(cL[0].y - cL[1].y) * debugY;
	int dR = (cR[1].x - cR[0].x)/(cR[0].y - cR[1].y) * debugY;

	//cL.pop_back();
	//cR.pop_back();

	for(int i = 0; i < 4; i++)
	{
		Point lL = cL[cL.size()-1];
		Point lR = cR[cR.size()-1];

		float maxS = 0;
		float scoreL[2*rangeX + 1];
		float scoreR[2*rangeX + 1];

		float maxL = 0;
		float maxR = 0;
		float minL = 9000000;
		float minR = 9000000;


		float histL[200];
		float histR[200];

		float histLs[200];
		float histRs[200];

		int hstRange = 40;

		Point lh1(-hstRange+lL.x, lL.y - debugY);
		Point lh2(hstRange+lL.x, lL.y);

		Point rh1(-hstRange+lR.x, lR.y - debugY);
		Point rh2(hstRange+lR.x, lR.y);

		calcHistogram(lh1, lh2, mProbVal, mProbBin, histL);
		calcHistogram(rh1, rh2, mProbVal, mProbBin, histR);

		line(filteredDbg, lh1, Point(lh1.x, lh2.y), CvScalar(0, 0, 127), 2);
		line(filteredDbg, lh2, Point(lh2.x, lh1.y), CvScalar(0, 0, 127), 2);
		line(filteredDbg, rh1, Point(rh1.x, rh2.y), CvScalar(0, 0, 127), 2);
		line(filteredDbg, rh2, Point(rh2.x, rh1.y), CvScalar(0, 0, 127), 2);



		for (int i =0; i < 200; i++)
		{
			histLs[i] = 0;
			histRs[i] = 0;
		}

		for (int i = 20; i < histRange*2-20; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				histLs[i] += histL[i+j];
				histRs[i] += histR[i+j];
			}
			histLs[i] += histL[i];
			histRs[i] += histR[i];
		}

		for (int j = -rangeX; j <= rangeX; j++)
		{
			float l = 0;
			float r = 0;

			if (debugZ == 1)
			{
				l = histLs[j+histRange];
				r = histRs[j+histRange];
			}
			else if (debugZ == 2)
			{
				for (int shift = -1; shift < 1; shift++)
				{
					l += calcScore(mProbVal, Point(lL.x - shift, lL.y), Point(lL.x - shift + j, lL.y - debugY));
					r += calcScore(mProbVal, Point(lR.x + shift, lR.y), Point(lR.x + shift + j, lR.y - debugY));
				}
			}

			scoreL[j+rangeX] = l;
			scoreR[j+rangeX] = r;

			if (maxL < l) maxL = l;
			if (maxR < r) maxR = r;

			if (minL > l) minL = l;
			if (minR > r) minR = r;
		}

		maxL /=100;
		maxR /=100;

		for (int j = -rangeX; j <= rangeX; j++)
		{
			float sc = (scoreL[j+rangeX])/ maxL;
			line(filteredDbg, Point2f(lL.x + j, y), Point2f(lL.x + j, y-sc), CvScalar(0, sc*5, sc*5), 2);

			sc = (scoreR[j+rangeX]) / maxR;
			line(filteredDbg, Point2f(lR.x + j, y), Point2f(lR.x + j, y-sc), CvScalar(0, sc*5, 0), 2);
		}

		maxS = 0;

		int newdL;
		int newdR;
		for (int iL = -rangeX; iL <= rangeX; iL++)
		{
			for (int iR = -rangeX; iR <= rangeX; iR++)
			{
				float score = scoreL[iL + rangeX] * scoreR[iR + rangeX];

				float centerChangeCm = abs((iR + iL) - (dR + dL))/pixelsPerCm/2;
				float widthChangeCm  = abs(iR - iL)/pixelsPerCm;

				float coef1 = 1.0 - centerChangeCm / (centerChangeCm + 50.0);
				float coef2 = 1.0 - widthChangeCm / (widthChangeCm + 40.0);
				score *= coef1 * coef2;

				//float lWidth_cm = (lR.x + iR - lL.x - iL)/pixelsPerCm;

			    if (/*(mLaneFilter->LANE.MIN_WIDTH <= lWidth_cm && lWidth_cm <= mLaneFilter->LANE.MAX_WIDTH) && */(score > maxS))
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

for (int shift = -20; shift < 20; shift++)
{
	//cout << shift << "\t" << calcGradScore(cL[cL.size()-2]+Point2f(shift, 0), cL[cL.size()-1]+Point2f(shift, 0)) << "\n";
}


	drawPointsX(filteredDbg, cL);
	drawPointsX(filteredDbg, cR);
	if (debugX == 0) imshow("filteredDbg", filteredDbg);


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
