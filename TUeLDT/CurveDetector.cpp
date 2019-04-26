/*
 * CurveDetector.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#include "CurveDetector.h"
#include "opencv2/opencv.hpp"
#include "BirdView.h"


using namespace cv;

CurveDetector::CurveDetector(const LaneTracker::Config* cfg,
		const LaneFilter* laneFilter,
		const VanishingPtFilter* vpFilter,
		const Templates* templates) :
		mCfg(cfg),
		mLaneFilter(laneFilter),
		mVpFilter(vpFilter),
		mTemplates(templates){
	// TODO Auto-generated constructor stub
	bufIt = 0;

}

CurveDetector::~CurveDetector() {
	// TODO Auto-generated destructor stub
}


void CurveDetector::prepareROI(cv::UMat input, cv::UMat& output)
{
	int span = (mCfg->cam_res_v/2) - mTemplates->HORIZON_ICCS_V + mVpFilter->RANGE_V;
	mROIy = mCfg->cam_res_v - span;
	assert((mROIy + span) == input.rows);
	cv::Rect lROI = cv::Rect(0, mROIy, mCfg->cam_res_h, span);
	input(lROI).copyTo(output);
	cerr << "ROI height = " << output.rows << "\n";
}


void CurveDetector::blur(cv::UMat input, cv::UMat& output)
{
	//input.copyTo(output);

	GaussianBlur(input, output, cv::Size(3,3), 2, 2);
}

int CurveDetector::TIPPING_POINT_GRAY = 200;
int CurveDetector::TIPPING_POINT_GRAD_Mag = 50;

/*
void CurveDetector::computeMap(cv::Mat& input, cv::Mat& outputMag, cv::Mat& outputAng)
{
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

    mMask = mGradX ==0;
    mGradX.setTo(1, mMask);

	divide(mGradY, mGradX, mAngMap, 256, CV_16S);


	//convert to absolute scale and add weighted absolute gradients
	absdiff(mGradX, cv::Scalar::all(0), mGradX_abs);
	absdiff(mGradY, cv::Scalar::all(0), mGradY_abs);

	add(mGradX_abs, mGradY_abs, tmp, noArray(), CV_8U);
	tmp -= TIPPING_POINT_GRAD_Mag;
	tmp.copyTo(tmp2);
	tmp2 += 10;
	divide(tmp, tmp2, mProbMap_GradMag, 255, CV_8U);


	input.convertTo(tmp, CV_8U);
	tmp -= TIPPING_POINT_GRAY;
	tmp.copyTo(tmp2);
	tmp2 += 10;
	divide(tmp, tmp2, mProbMap_Gray, 255, CV_8U);

	addWeighted(mProbMap_GradMag, 1, mProbMap_Gray, 0, 0, mProbVal, CV_8U);


	mProbVal.copyTo(outputMag);
	mAngMap.copyTo(outputAng);
}


*/



void CurveDetector::computeMap(cv::Mat& input, cv::Mat& outputMag, cv::Mat& outputAng)
{
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

    mMask = mGradX ==0;
    mGradX.setTo(1, mMask);

	divide(mGradY, mGradX, mAngMap, 256, CV_16S);


	//convert to absolute scale and add weighted absolute gradients
	absdiff(mGradX, cv::Scalar::all(0), mGradX_abs);
	absdiff(mGradY, cv::Scalar::all(0), mGradY_abs);

	add(mGradX_abs, mGradY_abs, tmp, noArray(), CV_8U);
	tmp -= TIPPING_POINT_GRAD_Mag;
	tmp.copyTo(tmp2);
	tmp2 += 10;
	divide(tmp, tmp2, mProbMap_GradMag, 255, CV_8U);


	input.convertTo(tmp, CV_8U);
	tmp -= TIPPING_POINT_GRAY;
	tmp.copyTo(tmp2);
	tmp2 += 10;
	divide(tmp, tmp2, mProbMap_Gray, 255, CV_8U);

	addWeighted(mProbMap_GradMag, 1, mProbMap_Gray, 0, 0, mProbVal, CV_8U);


	mProbVal.copyTo(outputMag);
	mAngMap.copyTo(outputAng);
}



const int BIRD_WIDTH = 400;
const int BIRD_HEIGHT = 900;

void CurveDetector::setParams(LaneModel* Lane, Mat roi)
{
    int binWidth = mLaneFilter->LANE.AVG_WIDTH/mLaneFilter->BINS_STEP_cm;
    int zeroPos = mLaneFilter->BASE_BINS.rows/2;
    int lPos = zeroPos - binWidth/2;
    int rPos = zeroPos + binWidth/2;

    assert(lPos >= 0);
    assert(rPos < mLaneFilter->BASE_BINS.rows);

    defaultVp = mLaneFilter->O_ICCS_ICS;
    defaultVp += Point2f(mLaneFilter->CAMERA.HORIZON_VH[1], mLaneFilter->CAMERA.HORIZON_VH[0]);
    defaultVp.y += - mCfg->cam_res_v + roi.rows;

    baseL = defaultVp;
    baseL.y = roi.rows;
    baseL.x += mLaneFilter->BASE_BINS.at<int>(0, lPos);

    baseR = defaultVp;
    baseR.y = roi.rows;
    baseR.x += mLaneFilter->BASE_BINS.at<int>(0, rPos);

	mBird.configureTransform2(baseL, baseR, defaultVp, BIRD_WIDTH, BIRD_HEIGHT);


	r1.y = l1.y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - mCfg->cam_res_v + roi.rows;
	r2.y = l2.y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y - mCfg->cam_res_v  + roi.rows;

	r1.x = Lane->boundaryRight[0] + mLaneFilter->O_ICCS_ICS.x;
	r2.x = Lane->boundaryRight[1] + mLaneFilter->O_ICCS_ICS.x;

	l1.x = Lane->boundaryLeft[0] + mLaneFilter->O_ICCS_ICS.x;
	l2.x = Lane->boundaryLeft[1] + mLaneFilter->O_ICCS_ICS.x;


	vector<Point2f> tmp, tmp2;
	tmp.push_back(r1);
	tmp.push_back(l1);
	tmp.push_back(r2);
	tmp.push_back(l2);
	mBird.convertPointsToBird(tmp, tmp2);
	assert(tmp2.size() == 4);
	birdRightStart = tmp2[0];
	birdLeftStart = tmp2[1];

	rightB = -(tmp2[0].x - tmp2[2].x) / (tmp2[0].y - tmp2[2].y);
	leftB = -(tmp2[1].x - tmp2[3].x) / (tmp2[1].y - tmp2[3].y);


	cerr << tmp2[0] << tmp2[2] << leftB << "\t" << rightB << endl;

	cerr << r1 << l1 << endl;
}




vector<int> curve(double a, double b, double c)
{
	vector<int> ret;

	for (int i = 0; i < BIRD_HEIGHT; i++)
	{
		int y = BIRD_HEIGHT-i;
		double val = 10e-5 * a * y * y + b * y + c;
		ret.push_back((int)val);
	}
	return ret;
}


vector<int> dif(double a, double b, double c)
{
	vector<int> ret;

	for (int i = 0; i < BIRD_HEIGHT; i++)
	{
		int y = BIRD_HEIGHT-i;
		double val = 2.0 * 10e-5 * a * y +  b;
		val *= 256;
		ret.push_back((int)val);
		//cerr << "dif \t" << a << "\t" << i << "\t" << (int)val << endl;
	}
	return ret;
}

float score(vector<int> & from, vector<int> & to, vector<int> & fromD, vector<int> & toD, Mat img, Mat ang)
{
	assert(from.size() == to.size());

	int ret = 0;
	int cnt = 1;
	for (int y = BIRD_HEIGHT/2; y < from.size(); y++)
	{
		int fromX = from[y] - 20;
		if (fromX < 0) fromX = 0;
		int toX = to[y] + 20;
		if (toX > img.cols-1) toX = img.cols - 1;

		for (int x = fromX; x <= toX; x++)
		{
			int val = (int)img.at<unsigned char>(y, x);
			int angVal = (int)ang.at<short int>(y, x);
			//if (val > 100) cerr << x << "\t" << y << "\t" << val << "\t" << angVal << "\t" << fromD[y] << "\t" << toD[y] << "\n";
			//if ((fromD[y] < angVal) && (angVal < toD[y])) ret += val;
			float dif = abs(fromD[y] - angVal);
			float angleProbability = 1 - dif / (dif + 0.1);
			angleProbability *= 100.0;
			ret += val * angleProbability;


			cnt ++;

		}
	}

	return (float)ret/cnt;
}



int CurveDetector::run(cv::UMat& frame, LaneModel* Lane)
{
	if (mCfg->display_graphics)  imshow("curveDet", frame);
	Point mO_ICCS_ICS(frame.cols/2, frame.rows/2);

	mInput = frame;

	prepareROI(mInput, mROI);
	blur(mROI, mROIblurred);
	if (mCfg->display_graphics) imshow("ROI", mROIblurred);
	cv::Mat tmp;
	tmp = mROIblurred.getMat(ACCESS_READ);

	Mat birdRaw;
	setParams(Lane, tmp);
	birdRaw = mBird.applyTransformation(tmp);
	if (mCfg->display_graphics) imshow("birdRaw", birdRaw);
	computeMap(birdRaw, mFrMag, mFrAng);
	if (bufMag[0].rows == 0)
	{
		for (int i = 0; i < 10; i++)
		{
			mFrMag.copyTo(bufMag[i]);
			mFrAng.copyTo(bufAng[i]);
		}
	}

	mFrMag.copyTo(bufMag[bufIt]);
	mFrAng.copyTo(bufAng[bufIt]);
	bufIt = (bufIt+1)%3;


	for (int i = 0; i < 3; i++)
	{
		mMask = mFrMag < bufMag[i];

		bufMag[i].copyTo(mFrMag, mMask );
		bufAng[i].copyTo(mFrAng, mMask );
	}




	if (mCfg->display_graphics) imshow("mFrMag", mFrMag);



	Mat debugFrame;
	mFrMag.copyTo(debugFrame);
    cvtColor(debugFrame, debugFrame, COLOR_GRAY2BGR);


    line(debugFrame, Point(birdLeftStart + Point2f(2, 2)), Point(birdLeftStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdLeftStart + Point2f(-2, 2)), Point(birdLeftStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);

    line(debugFrame, Point(birdRightStart + Point2f(2, 2)), Point(birdRightStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdRightStart + Point2f(-2, 2)), Point(birdRightStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);



    float maxI;
    float maxScore = 0;
	for (float i = -1; i < 1; i+=0.1)
	{
		float MARGIN = 0.15;
		vector<int> from, to, fromD, toD, test;

		from = curve(i - MARGIN, rightB, birdRightStart.x);
		to = curve(i + MARGIN, rightB, birdRightStart.x);

		fromD = dif(i - MARGIN, rightB, birdRightStart.x);
		toD = dif(i + MARGIN, rightB, birdRightStart.x);


		float sR = score(from, to, fromD, toD, mFrMag, mFrAng);

		test  = curve(i, rightB, birdRightStart.x);
		for (int i = 1; i < test.size(); i++)
		{
			//line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(sR*12, 0, 0), 1);
		}


		from = curve(i - MARGIN, leftB, birdLeftStart.x);
		to = curve(i + MARGIN, leftB, birdLeftStart.x);

		fromD = dif(i - MARGIN, leftB, birdLeftStart.x);
		toD = dif(i + MARGIN, leftB, birdLeftStart.x);


		float sL = score(from, to, fromD, toD, mFrMag, mFrAng);

		test  = curve(i, leftB, birdLeftStart.x);
		for (int i = 1; i < test.size(); i++)
		{
			//line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(sL*12,0, 0), 1);
		}

		float score = sL * sL;

		//cerr << i << "\t" << score << endl;

		line(debugFrame, Point(i*10 + 100, BIRD_HEIGHT), Point(i*10 +100, BIRD_HEIGHT-score/100), CvScalar(0,0, 255), 1);
		if (maxScore < score)
		{
			maxScore = score;
			maxI = i;
		}
	}

	vector<int> test  = curve(maxI, leftB, birdLeftStart.x);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,255, 0), 2);
	}
	Lane->curveL.clear();

	vector<Point2f> out;
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}

	mBird.invertPoints(out, Lane->curveL);
	out.clear();

	test  = curve(maxI, rightB, birdRightStart.x);

	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}

	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,255, 0), 2);
	}

	mBird.invertPoints(out, Lane->curveR);

	for (Point2f & pt : Lane->curveL) pt += Point2f(0, mROIy);
	for (Point2f & pt : Lane->curveR) pt += Point2f(0, mROIy);

	if (mCfg->display_graphics)  imshow("debugFrame", debugFrame);



	return 0;
}
