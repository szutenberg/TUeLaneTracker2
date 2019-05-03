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
}


int CurveDetector::TIPPING_POINT_GRAY = 200;
int CurveDetector::TIPPING_POINT_GRAD_Mag = 50;

void CurveDetector::computeMap(cv::Mat& input, cv::Mat& outputMag, cv::Mat& outputAng)
{
	Sobel( input, mGradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

	for (int i = 1; i < segmentsToRemove.size(); i++)
	{
		line(mGradY, segmentsToRemove[i-1], segmentsToRemove[i], cv::Scalar(0), 8);
		line(mGradX, segmentsToRemove[i-1], segmentsToRemove[i], cv::Scalar(0), 8);

	}

    mMask = mGradX ==0;
    mGradX.setTo(1, mMask);

	divide(mGradY, mGradX, mAngMap, 128, CV_16S);

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

	addWeighted(mProbMap_GradMag, 0.8, mProbMap_Gray, 0.2, 0, mProbVal, CV_8U);

	mProbVal.copyTo(outputMag);
	mAngMap.copyTo(outputAng);
}



const int BIRD_WIDTH = 350;
const int BIRD_HEIGHT = 1000;

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
	tmp.push_back(Point(0, defaultVp.y + 50));
	tmp.push_back(Point(0, roi.rows));
	tmp.push_back(Point(roi.cols, roi.rows));
	tmp.push_back(Point(roi.cols, defaultVp.y + 50));

	mBird.convertPointsToBird(tmp, tmp2);
	assert(tmp2.size() == 8);

	segmentsToRemove.clear();
	segmentsToRemove.push_back(tmp2[4]);
	segmentsToRemove.push_back(tmp2[5]);
	segmentsToRemove.push_back(tmp2[6]);
	segmentsToRemove.push_back(tmp2[7]);

	birdRightStart = tmp2[0];
	birdLeftStart = tmp2[1];

	rightB = -(tmp2[0].x - tmp2[2].x) / (tmp2[0].y - tmp2[2].y);
	leftB = -(tmp2[1].x - tmp2[3].x) / (tmp2[1].y - tmp2[3].y);
}





Mat tmp;
double maxA, maxB;


int RANGE_FROM = -100;
int RANGE_TO = 100;
int RANGE_N = RANGE_TO - RANGE_FROM;
int STRIPES_AMOUNT = 5;
int STRIP_H = BIRD_HEIGHT/STRIPES_AMOUNT;
int CHART_H = 100;


vector<int> curve(double a, double b, double c)
{
	vector<int> ret;

	for (int i = 0; i < BIRD_HEIGHT; i++)
	{
		double y = BIRD_HEIGHT-i;
		double val = a * y * y + b * y + c;
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
		double val = 2.0 * 10e-5 * a * y +  b / 200;
		val *= 256;
		ret.push_back((int)val);
		//cerr << "dif \t" << a << "\t" << i << "\t" << (int)val << endl;
	}
	return ret;
}

void calculateHistograms(cv::Mat img, cv::Mat ang, double* vals)
{
	Mat mask, strip;
	cv::Rect lROI;

	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		double maxV = 0.01;
		double total = 0;
		lROI = cv::Rect(0, STRIP_H*s, img.cols, STRIP_H);

		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			img(lROI).copyTo(strip);
			mask = ang(lROI) != i;
			strip.setTo(0, mask);
			double val = cv::sum( strip )[0];
			vals[s*RANGE_N + i-RANGE_FROM] = val;
			if (maxV < val) maxV = val;
			total += val;
		}

		//cerr << "d " << s << "\t" << maxV << "\t" << total/maxV << "\n";
		maxV += total/RANGE_N;
		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			vals[s*RANGE_N + i-RANGE_FROM] += total/RANGE_N;
			vals[s*RANGE_N + i-RANGE_FROM] /= maxV;
		}
	}
}


void plotHistograms(const char * name, double* vals)
{
	Mat plot = Mat::zeros(STRIPES_AMOUNT*CHART_H, RANGE_N, CV_8U);
	line(plot, Point(-RANGE_FROM, plot.rows ), Point(-RANGE_FROM, 0), cv::Scalar(127));

	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			double val = vals[s*RANGE_N + i-RANGE_FROM];
			line(plot, Point(i-RANGE_FROM, CHART_H+s*CHART_H), Point(i-RANGE_FROM, CHART_H+s*CHART_H-val*CHART_H), cv::Scalar(val*255));
		}
	}
	imshow(name, plot);
}


void filterHistorgrams(double* in, double *out)
{
	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		for (int i = RANGE_FROM; i < RANGE_TO; i++) out[s * RANGE_N + i-RANGE_FROM] = 0;

		double maxV = 0.01;

		for (int i = RANGE_FROM + 4; i < RANGE_TO-4; i++)
		{
			double val = 0;
			for (int j = RANGE_FROM; j < RANGE_TO; j++)
			{
				val += in[s * RANGE_N + j - RANGE_FROM] * (1.0 - (double)abs(i-j)/((double)abs(i-j) + 0.1));
			}

			out[s * RANGE_N + i - RANGE_FROM] = val;
			if (maxV < val) maxV = val;
		}

		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			out[s * RANGE_N + i - RANGE_FROM] /= maxV;
		}
	}
}

void plotArray(const char * name, double *in, int N)
{
	int HEIGHT = 200;

	Mat plot = Mat::zeros(HEIGHT, N, CV_8U);

	for (int i = 0; i < N; i++)
	{
		if (in[i] >= 0) line(plot, Point(i, HEIGHT), Point(i, HEIGHT - HEIGHT*in[i]), cv::Scalar(255), 1);
		else line(plot, Point(i, HEIGHT), Point(i, HEIGHT + HEIGHT*in[i]), cv::Scalar(127), 1);
	}
	imshow(name, plot);
}


double calcHistProb(double *in, double a, double b)
{
	double prob = 1;
	double minP = 1;

	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		int offset = (STRIPES_AMOUNT - s - 1) * RANGE_N;
		int index = a * (s + 0.5) * STRIP_H + b * STRIP_H - RANGE_FROM + 0.5;
		double val = 0.1;

		if ((index > 1) && (index < RANGE_N-1))
		{
			val = 0.25 * in[offset + index - 1] + 0.5 *in[offset + index] + 0.25 * in[offset + index - 1];
		}

		prob *= val;
		if (minP > val) minP = val;
	}

	if (minP > 0) prob /= minP;

	return prob;
}


void findCandidates(double *in, double *params, int* amt)
{
	*amt = 0;

	double arg[2000];
	double t[2000];
	double t2[2000];
	double t3[2000];
	double t4[2000];

	int N = 0;

	for (double a = -0.1; a <= 0.1; a+=0.001)
	{
		double maxProb = 0;
		for (double b = -0.5; b < 0.5; b+=0.01)
		{
			double prob = calcHistProb(in, a, b);
			if (maxProb < prob) maxProb = prob;
		}
		t2[N] = 0;
		t3[N] = 0;
		arg[N] = a;
		t[N++] = maxProb;
	}

	plotArray("a", t, N);

	double maxVal = 0;
	int candidate1 = 0;
	int candidate2 = 0;

	for (int i = 5; i < N-5; i++)
	{
		double val = t[i-2] + t[i-1] * 2.0 + t[i] * 4.0 + t[i+1] * 2.0 + t[i+2];
		val /= 10.0;
		t2[i] = val;
		if (maxVal < val) maxVal = val, candidate1 = i;
	}

	t2[candidate1] *= -1;

	plotArray("t2", t2, N);

	maxVal = 0;
	for (int i = 0; i < N; i++)
	{
		double d = abs(i - candidate1);
		double val = t2[i] * (d / (d + 30.0));
		t3[i] = val;

		if (maxVal < val) maxVal = val, candidate2 = i;
	}

	t3[candidate2] *= -1;



	plotArray("t3", t3, N);
	t[candidate1] *= -1;
	t[candidate2] *= -1;


	cerr << "\t" << arg[candidate1] << "\t" << arg[candidate2] << endl;

	plotArray("selected", t, N);


	double candidates[2] = {arg[candidate1], arg[candidate2]};

	N = 0;
	double canB[2];

	for (int cand = 0; cand < 2; cand++)
	{
		double a = candidates[cand];

		double maxP = 0;
		double maxB = 0;
		for (double b = -0.5; b < 0.5; b+=0.005)
		{
			double prob = calcHistProb(in, a, b);
			t4[N++] = prob;
			if (maxP < prob) maxP=prob, maxB=b;
		}
		canB[cand] = maxB;
	}



	plotArray("b", t4, N);
	params[0] = arg[candidate1] / STRIP_H;
	params[1] = canB[0];
	params[2] = arg[candidate2] / STRIP_H;
	params[3] = canB[1];
	*amt = 2;
}



void calculateMatch(double *dest, int N, double a, double b, Mat img, Mat ang)
{
	Mat mask, imgRoi, angRoi, difRoi, tmp, imgRoi2, difRoi2;
	Mat img2;

	img.copyTo(img2);
	cv::Rect lROI;

	for (int i = 0; i < N; i++) dest[i] = 0;

	int STR_N = 10;
	int STR_H = img.rows / STR_N;

	cerr << a << "\t" << b << endl;
	for (int s = 0; s < STR_N; s++)
	{
		lROI = cv::Rect(0, STR_H*s, img.cols, STR_H);

		img(lROI).copyTo(imgRoi);
		ang(lROI).copyTo(angRoi);



		double y = BIRD_HEIGHT-((double)s+0.5)*(double)STR_H;

		double dif = a * y * y + b * y;
		double angle = (2 * a * y + b) * 128;
		angRoi.copyTo(difRoi);
		difRoi -= angle;
		difRoi = abs(difRoi);
		tmp = difRoi + 4;
		difRoi = tmp - difRoi;

		divide(difRoi, tmp, difRoi2, 1, CV_32F);
		multiply(difRoi2, imgRoi, imgRoi2, 1, CV_8U);

		Mat dst;

		reduce(imgRoi2, dst, 0, CV_REDUCE_SUM, CV_32S);

		int * ptr = dst.ptr<int>(0);
		for (int i = 0; i < dst.cols; i++)
		{
			int index = i - dif;
			if ((index >= 0) && (index < N)) dest[index] += ptr[i];
		}

		//imgRoi2.copyTo(img2(Rect(0, STR_H*s, imgRoi.cols, imgRoi.rows)));
	}

	//imshow("img2", img2);

}


void CurveDetector::bufferFiltering(int bufferSize)
{
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
	bufIt = (bufIt+1)%bufferSize;

	for (int i = 0; i < bufferSize; i++)
	{
		mMask = mFrMag < bufMag[i];

		bufMag[i].copyTo(mFrMag, mMask );
		bufAng[i].copyTo(mFrAng, mMask );
	}
}

int CurveDetector::run(cv::UMat& frame, LaneModel* Lane)
{
	Mat birdRaw, tmp;

	Point mO_ICCS_ICS(frame.cols/2, frame.rows/2);
	mInput = frame;

	prepareROI(mInput, mROI);

	GaussianBlur(mROI, mROIblurred, cv::Size(3,3), 3, 3);

	if (mCfg->display_graphics) imshow("mROIblurred", mROIblurred);

	tmp = mROIblurred.getMat(ACCESS_READ);

	setParams(Lane, tmp);

	birdRaw = mBird.applyTransformation(tmp);

	if (mCfg->display_graphics) imshow("birdRaw", birdRaw);

	GaussianBlur(birdRaw, birdRaw, cv::Size(5, 5), 2, 2);

	computeMap(birdRaw, mFrMag, mFrAng);

	bufferFiltering(mCfg->buffer_count);

	if (mCfg->display_graphics) imshow("mFrMag", mFrMag);


	Mat debugFrame;
	mFrMag.copyTo(debugFrame);
    cvtColor(debugFrame, debugFrame, COLOR_GRAY2BGR);

    line(debugFrame, Point(birdLeftStart + Point2f(2, 2)), Point(birdLeftStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdLeftStart + Point2f(-2, 2)), Point(birdLeftStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);

    line(debugFrame, Point(birdRightStart + Point2f(2, 2)), Point(birdRightStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdRightStart + Point2f(-2, 2)), Point(birdRightStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);

  //  score2(mFrMag, mFrAng, 0);

    double hist[STRIPES_AMOUNT*RANGE_N], hist2[STRIPES_AMOUNT*RANGE_N];
    double hist3[STRIPES_AMOUNT*RANGE_N];
    calculateHistograms(mFrMag, mFrAng, &hist[0]);
    plotHistograms("before Filtering", &hist[0]);
    filterHistorgrams(&hist[0], &hist2[0]);
    plotHistograms("after Filtering", &hist2[0]);


    double params[10];
    int candidatesN;
    findCandidates(&hist2[0], &params[0], &candidatesN);

    double cHists[10000], histNorm[10000];
    calculateMatch(&cHists[0], mFrMag.cols, params[0], params[1], mFrMag, mFrAng);
    calculateMatch(&cHists[mFrMag.cols], mFrMag.cols, params[2], params[3], mFrMag, mFrAng);


    for (int n = 0; n < candidatesN; n++)
    {
    	int offset = n * mFrMag.cols;
    	for (int i = 0; i < mFrMag.cols; i++)
    	{
    		double dL = abs(i - birdLeftStart.x);
    		double weight1 = 1 - dL / (dL + 1);

    		double dR = abs(i - birdRightStart.x);
    		double weight2 = 1 - dR / (dR + 1);

    		hist3[offset + i] = (weight1 + weight2) * cHists[offset + i];
    	}
    }


    double maxV = 0;
    for (int i = 0; i < mFrMag.cols*candidatesN; i++) if (maxV < cHists[i]) maxV = cHists[i];
    for (int i = 0; i < mFrMag.cols*candidatesN; i++) cHists[i] /= maxV;

	plotArray("c", &cHists[0], mFrMag.cols*candidatesN);

	maxV = 0;
	for (int i = 0; i < mFrMag.cols*candidatesN; i++) if (maxV < hist3[i]) maxV = hist3[i];
	for (int i = 0; i < mFrMag.cols*candidatesN; i++) hist3[i] /= maxV;

	plotArray("hist3 c", &hist3[0], mFrMag.cols*candidatesN);

	double bestSum = -1;
	double bestA, bestB;
	for (int c = 0; c < candidatesN; c++)
	{
		double sum = 0;
		int offset = c * mFrMag.cols;
		for (int i = 0; i < mFrMag.cols; i++)
		{
			sum += hist3[offset + i];
		}
		cerr << c << " got " << sum << "\n";

		if (bestSum < sum) bestSum = sum, bestA = params[c*2], bestB = params[c*2 + 1];
	}



    rightB = (maxB - 100) / 128;
    leftB = (maxB - 100) / 128;

	vector<int> test  = curve(params[2], params[3], birdLeftStart.x);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,255, 0), 2);
	}

	test  = curve(params[0], params[1], birdLeftStart.x);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}
	Lane->curveL.clear();

	vector<Point2f> out;
	test  = curve(bestA, bestB, birdLeftStart.x);
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}

	mBird.invertPoints(out, Lane->curveL);
	out.clear();

	test  = curve(params[0], params[1], birdRightStart.x);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}

	test  = curve(params[2], params[3], birdRightStart.x);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,255, 0), 2);
	}

	test  = curve(bestA, bestB, birdRightStart.x);
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}
	mBird.invertPoints(out, Lane->curveR);

	for (Point2f & pt : Lane->curveL) pt += Point2f(0, mROIy);
	for (Point2f & pt : Lane->curveR) pt += Point2f(0, mROIy);

	if (mCfg->display_graphics)  imshow("debugFrame", debugFrame);



	return 0;
}
