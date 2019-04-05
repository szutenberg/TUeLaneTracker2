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


cv::Mat TrackingLaneDAG_generic::createProbabilityMap(cv::Mat input)
{
	// TODO use different variables when making parallel
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

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
	subtract(input, mLaneMembership.TIPPING_POINT_GRAY, mTempProbMat, cv::noArray(), CV_32S);
	mMask = mTempProbMat <0 ;
	mTempProbMat.setTo(0,mMask);
	mTempProbMat.copyTo(mProbMap_Gray);
	mTempProbMat = mTempProbMat + 10;

	divide(mProbMap_Gray, mTempProbMat, mProbMap_Gray, 255, -1);

	//GradientMag Probabilities
	subtract(mFrameGradMag, mLaneMembership.TIPPING_POINT_GRAD_Mag*0 + 20, mTempProbMat, cv::noArray(), CV_32S);
	mTempProbMat.copyTo(mProbMap_GradMag);
	mTempProbMat= abs(mTempProbMat) + 10;
	divide(mProbMap_GradMag, mTempProbMat, mProbMap_GradMag, 255, -1);

	// Intermediate Probability Map
	mBufferPool->Probability[mBufferPos] = mProbMap_GradMag + mProbMap_Gray;
	mMask = mBufferPool->Probability[mBufferPos] <0 ;
	mBufferPool->Probability[mBufferPos].setTo(0,mMask);
	mBufferPool->Probability[mBufferPos].copyTo(mProbMapNoTangent);

	//Gradient Tangent Probability Map
	//	subtract(mGradTanTemplatescore, mBufferPool->GradientTangent[mBufferPos], mTempProbMat, cv::noArray(), CV_32S);
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

	return mBufferPool->Probability[mBufferPos];
}

cv::Mat TrackingLaneDAG_generic::createHistogram(cv::Mat input)
{
	// TODO use different variables when making parallel
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

	//convert to absolute scale and add weighted absolute gradients
	mGradX_abs = abs(mGradX);
	mGradY_abs = abs(mGradY );

	//addWeighted( mGradX_abs, 0.5, mGradY_abs, 0.5, 0, mFrameGradMag );
	mFrameGradMag = mGradX_abs + mGradY_abs;

	//convertScaleAbs(mFrameGradMag, mFrameGradMag);
	mFrameGradMag.convertTo(mFrameGradMag, CV_8U);

	cv::divide(mGradY, mGradX, mBufferPool->GradientTangent[mBufferPos], 128, -1);

	int GRID = 40;

	int histWidth = input.cols/GRID;
	int histHeight = input.rows/GRID;

	float angle[histWidth][histHeight];
	float value[histWidth][histHeight];

	Mat dbg;
	input.copyTo(dbg);
    cvtColor(dbg, dbg, COLOR_GRAY2BGR);





	for (int hst = 0; hst < histWidth*histHeight; hst++)
	{

		int px = hst % histWidth;
		int py = hst / histWidth;
		angle[px][py] = 0;
		value[px][py] = 0;
		px *= GRID;
		py *= GRID;

		float mag = 0;
		float sum = 0;

		//cout << px << "\t" << py << "\n";

		float anV[181];
		for (int i = 0; i < 181; i++) anV[i] = 0;
		for (int i = px; i < px+GRID; i++)
		{
			for (int j = py; j < py+GRID; j++)
			{
				float gy = (int)mGradY.at<short>(j, i);
				float gx = (int)mGradX.at<short>(j, i);
				float mMag = abs(gy) + abs(gx);
				//if (mMag > 255) mMag = 255;
				if (mMag < 20) continue;
				int mAng = atan(gy / gx) * 180.0 / 3.14;
				assert(mAng >= -90);
				assert(mAng <= 90);

				anV[mAng + 90] += mMag;
				mag += mMag;

				//mag += abs(gy) + abs(gx);

				//cout << ang << "\t" << mag << "\n";
			}
		}

		if (mag < 1) continue;

		int ang;
		for (ang = 0; ang < 181; ang++)
		{
			sum += anV[ang];
			if ((sum * 2) > mag) break;
		}

		ang -= 90;

		//cout << mag << "\t" << sum << "\t" <<  sum/mag << "\n" ;
		float t = sin(ang*3.14/180.0);
		float c = cos(ang*3.14/180.0);

		line(dbg, Point(px+GRID/2, py+GRID/2), Point(px+GRID/2 - t*GRID, py+GRID/2 + c*GRID), CvScalar(0, 0, mag/100), 2);
		line(dbg, Point(px+GRID/2, py+GRID/2), Point(px+GRID/2 + t*GRID, py+GRID/2 - c*GRID), CvScalar(0, 0, mag/100), 2);



	}

	if (debugX == 0) imshow("dbg", dbg);



	return input;




}


void TrackingLaneDAG_generic::trackCurves2(cv::Mat& input)
{
	cv::Point r1, r2, l1, l2;
	BirdView bird;
	Mat birdRaw;
	Mat buffered;
	Mat prob; // probability map

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
    baseL.y = r1.y;
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
    //cout << mLaneFilter->BASE_LINE_cm << "\t" << mLaneFilter->LANE.AVG_WIDTH << endl;
    //cout << "we have " << mLaneFilter->LANE.AVG_WIDTH/mLaneFilter->BINS_STEP_cm << " bins\n";
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
	//createHistogram(birdRaw);
	prob = createProbabilityMap(birdRaw);

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

	line(prob, startPointsBird[4],startPointsBird[5], CvScalar(0), 5);
	line(prob, startPointsBird[6],startPointsBird[7], CvScalar(0), 5);

	if (debugX == 0) imshow("prob", prob);



	vector<Point2f> cL, cR;
	cR.push_back(startPointsBird[0]);
	Point2f avgP = (startPointsBird[0] + startPointsBird[1])/2;
	cR.push_back(avgP);

	cL.push_back(startPointsBird[2]);
	avgP = (startPointsBird[2] + startPointsBird[3])/2;
	cL.push_back(avgP);



	mBUF_SIZE = 5;


	if (mBuf[0].rows == 0)
	{
		for (int i = 0; i < mBUF_SIZE; i++) prob.copyTo(mBuf[i]);
		mBufIt = 0;
	}
	prob.copyTo(mBuf[mBufIt]);
	mBufIt = (mBufIt+1)%mBUF_SIZE;

	buffered = Mat::zeros(prob.rows, prob.cols, CV_8U);

	for (int i = 0; i < mBUF_SIZE; i++)
		addWeighted(buffered, 1, mBuf[i], 1.0/mBUF_SIZE, 1, buffered);

	//GaussianBlur(buffered, buffered, cv::Size(15,5), 0, 0);



	Mat filteredDbg;
	buffered.copyTo(filteredDbg);
    cvtColor(filteredDbg, filteredDbg, COLOR_GRAY2BGR);


	if (debugY == 0) debugY = 180;

	int y = cL[1].y - debugY;
	int rangeX = 60;

	int dL = (cL[1].x - cL[0].x)/(cL[0].y - cL[1].y) * debugY;
	int dR = (cR[1].x - cR[0].x)/(cR[0].y - cR[1].y) * debugY;

	for(int i = 0; i < cL[1].y/debugY - 1; i++)
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

		for (int j = -rangeX; j <= rangeX; j++)
		{
			float l = 0;
			float r = 0;

			for (int shift = -4; shift <= 4; shift++)
			{
				l += calcScore(buffered, lL+Point(shift, 0), Point2f(lL.x + j + shift, y));
				r += calcScore(buffered, lR+Point(shift, 0), Point2f(lR.x + j + shift, y));
			}
/*
			for (int shift = 6; shift <= 8; shift++)
			{
				l -= (calcScore(buffered, lL+Point(shift, 0), Point2f(lL.x + j + shift, y)))*0.5;
				r -= (calcScore(buffered, lR+Point(-shift, 0), Point2f(lR.x + j - shift, y)))*0.5;
			}*/

			scoreL[j+rangeX] = l;
			scoreR[j+rangeX] = r;

			if (maxL < l) maxL = l;
			if (maxR < r) maxR = r;

			if (minL > l) minL = l;
			if (minR > r) minR = r;
		}

		if ((maxR-1) < (minR)) minR -= 1;
		if ((maxL-1) < (minL)) minL -= 1;

		maxR -= minR;
		maxL -= minL;


		maxL /= 50;
		maxR /= 50;

		for (int j = -rangeX; j <= rangeX; j++)
		{
			float sc = (scoreL[j+rangeX] - minL)/ maxL;
			line(filteredDbg, Point2f(lL.x + j, y), Point2f(lL.x + j, y-sc), CvScalar(0, sc*5, sc*5), 2);

			sc = (scoreR[j+rangeX] - minR) / maxR;
			line(filteredDbg, Point2f(lR.x + j, y), Point2f(lR.x + j, y-sc), CvScalar(0, sc*5, 0), 2);
		}

		maxS = 0;

		int newdL;
		int newdR;
		for (int iL = -rangeX; iL <= rangeX; iL++)
		{
			for (int iR = -rangeX; iR <= rangeX; iR++)
			{
//				float laneWidthDifCm = abs(iL - iR)/pixelsPerCm;
				float score = scoreL[iL + rangeX] * scoreR[iR + rangeX];
/*
				float changeDirCm = (abs(dL - iL) + abs(dR - iR))/pixelsPerCm;
				if (((dL < iL) && (dR < iR)) || ((dL > iL) && (dR > iR)))
				{
					changeDirCm = (abs(dL - iL - (dR - iR)))/pixelsPerCm;
				}

*/

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



		//float lWidth_cm = (lR.x + dR - lL.x - dL)/pixelsPerCm;
		//int laneLoc = (lR.x + dL + lL.x + dR)/2;
		//cout << i << "\t" << lWidth_cm << "\t" << laneLoc << "\n";



		cL.push_back(Point2f(lL.x + dL, y));
		cR.push_back(Point2f(lR.x + dR, y));
		y -= debugY;
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
