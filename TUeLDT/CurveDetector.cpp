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
	//cerr << "ROI height = " << output.rows << "\n";
}


void CurveDetector::blur(cv::UMat input, cv::UMat& output)
{
	//input.copyTo(output);

	GaussianBlur(input, output, cv::Size(3,3), 3, 3);
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

	addWeighted(mProbMap_GradMag, 1, mProbMap_Gray, 0, 0, mProbVal, CV_8U);


	mProbVal.copyTo(outputMag);
	mAngMap.copyTo(outputAng);
}



const int BIRD_WIDTH = 250;
const int BIRD_HEIGHT = 500;

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


	//cerr << tmp2[0] << tmp2[2] << leftB << "\t" << rightB << endl;

	//cerr << r1 << l1 << endl;
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

Mat tmp;
Mat ddd;
double maxA, maxB;


int RANGE_FROM = -100;
int RANGE_TO = 100;
int RANGE_N = RANGE_TO - RANGE_FROM;
int STRIPES_AMOUNT = 5;
int STRIP_H = BIRD_HEIGHT/STRIPES_AMOUNT;
int CHART_H = 100;


float score2(Mat img, Mat ang, double a)
{
	ang.copyTo(tmp);
	img.copyTo(ddd);

	Mat mask;
	STRIP_H = img.rows/STRIPES_AMOUNT;
	Mat plot = Mat::zeros(STRIPES_AMOUNT*CHART_H, 800, CV_8U);

	double vals[STRIPES_AMOUNT][RANGE_TO - RANGE_FROM + 1];
	double maxV[STRIPES_AMOUNT];
	double valNorm[STRIPES_AMOUNT][RANGE_TO - RANGE_FROM + 1];

	for (int stripe = 0; stripe < STRIPES_AMOUNT; stripe++)
	{
		maxV[stripe] = 0;
		 cv::Rect lROI;

		 //Define ROI from the Input Image
		 lROI = cv::Rect(0, STRIP_H*stripe, img.cols, STRIP_H);
		 img(lROI).copyTo(ddd);

		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			img(lROI).copyTo(ddd);

			mask = ang(lROI) != i;
			ddd.setTo(0, mask);
			//bitwise_and(img, img, ddd, mask);
			double val = cv::sum( ddd )[0];
			vals[stripe][i-RANGE_FROM] = val;
			if (maxV[stripe] < val) maxV[stripe] = val;
		}
	}


	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		maxV[s] = 0;
		for (int i = RANGE_FROM; i < RANGE_TO; i++) valNorm[s][i-RANGE_FROM] = 0;

		for (int i = RANGE_FROM + 4; i < RANGE_TO-4; i++)
		{
			double val = 0.5 * vals[s][i-RANGE_FROM-2] + vals[s][i-RANGE_FROM-1] + 2.0 * vals[s][i-RANGE_FROM] + vals[s][i-RANGE_FROM+1] + 0.5 * vals[s][i-RANGE_FROM+2];
			//val += vals[s][i-RANGE_FROM-3] + vals[s][i-RANGE_FROM+3];
			//val += vals[s][i-RANGE_FROM-4] + vals[s][i-RANGE_FROM+4];

			valNorm[s][i-RANGE_FROM] = val;
			if (maxV[s] < val) maxV[s] = val;
		}
	}




	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		cerr << "Max : " << s << " " << maxV[s] << endl;
		for (int i = RANGE_FROM; i < RANGE_TO; i++) valNorm[s][i-RANGE_FROM] /= maxV[s];
	}

	line(plot, Point(-RANGE_FROM, plot.rows ), Point(-RANGE_FROM, 0), cv::Scalar(127));
	double maxProb = 0;

	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			double val = valNorm[s][i-RANGE_FROM];
			line(plot, Point(i-RANGE_FROM, CHART_H+s*CHART_H), Point(i-RANGE_FROM, CHART_H+s*CHART_H-val*CHART_H), cv::Scalar(val*255));
		}
	}


	for (double a = -5; a < 5; a += 0.1)
	{
		int dif[STRIPES_AMOUNT];

		for (int s = 0; s < STRIPES_AMOUNT; s++)
		{
			int y = BIRD_HEIGHT - (STRIP_H * 0.5 + STRIP_H * s);
			double val = 2.0 * 10e-5 * a * y * 128;
			dif[s] = (int)(val + 0.5);
			//cerr << "dif " << a << " (" << s << ")" << dif[s] << endl;
		}

		for (int b = RANGE_FROM; b < RANGE_TO; b++)
		{
			double prob = 1;

			for (int s = 0; s < STRIPES_AMOUNT; s++)
			{
				int loc = dif[s]+b;
				if ((loc > 10) && (loc < RANGE_N - 10))
				{
					double p = 0;

					for (int i = - 2; i <=  2; i++)
					{
						p += valNorm[s][loc + i] * (1 - abs(i) / (abs(i) + 1));
					}

					prob *= p;
				}
				else
				{
					prob = 0;
				}
			}

			//if (prob > 0.1) cerr << a << "\t" << b << "\t" << prob << "\n";
			if (prob > maxProb)
			{
				maxProb = prob;
				maxA = a;
				maxB = b;
			}
		}
	}

	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		int y = BIRD_HEIGHT - (STRIP_H * (s+1));
		double val = 2.0 * 10e-5 * maxA * y * 128;
		int dif = (int)(val + 0.5);

		//cerr << "dif " << a << " (" << s << ")" << dif[s] << endl;

		cerr << s << " " << maxA << " " << dif << endl;

		//line(plot, Point(maxB-RANGE_FROM, CHART_H+s*CHART_H), Point(maxB-RANGE_FROM, CHART_H+s*CHART_H-val*CHART_H), cv::Scalar(val*255));
		//line(plot, Point(maxB-RANGE_FROM, CHART_H+s*CHART_H), Point(maxB-RANGE_FROM, CHART_H+s*CHART_H-val*CHART_H), cv::Scalar(val*255));
	}






	cerr << maxA << "\t" << maxB << "\t" << maxProb << "\n";





	imshow("plot", plot);


}



void CurveDetector::matchParabolaWithMap(cv::Mat mFrMag, cv::Mat mFrAng,
		double maxA, double maxB, double values[], int N)
{
	for (int i = 0 ; i < N; i++) values[i] = i;

	vector<int> test  = curve(maxA, maxB, birdLeftStart.x);

	int s= 4;
	for (int y = 0; y < BIRD_HEIGHT; y++)
	{
		double val = 2.0 * 10e-5 * maxA * y * 128;
		cerr << y << "\t" << (int)(val + 0.5) << "\n";
	}
	Mat ddd;

	int stripe = STRIPES_AMOUNT - 1;
		 cv::Rect lROI;

		 //Define ROI from the Input Image
		 lROI = cv::Rect(0, STRIP_H*stripe, mFrMag.cols, STRIP_H);
		 mFrMag(lROI).copyTo(ddd);

		 Mat dst;
		 cerr << maxB << endl;
		 Mat mask = mFrAng(lROI) < (maxB-10);
		 ddd.setTo(0, mask);
		 mask = mFrAng(lROI) > (maxB+10);
		 ddd.setTo(0, mask);


		 reduce(ddd, dst, 0 /* single row*/, CV_REDUCE_AVG);

		 vector<float> vec;
		 dst.copyTo(vec);
		 assert(vec.size() >= N);
		 for (int i = 0; i < N; i++) values[i] = vec[i];



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
	GaussianBlur(birdRaw, birdRaw, cv::Size(5, 5), 2, 2);
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

    score2(mFrMag, mFrAng, 0);


    double values[debugFrame.cols];
    matchParabolaWithMap(mFrMag, mFrAng, maxA, maxB, &values[0], debugFrame.cols);

    for (int i = 0; i < debugFrame.cols; i++)
    {
		line(debugFrame, Point(i, debugFrame.rows), Point(i, debugFrame.rows - values[i]), CvScalar(0,0, 255), 2);
    }

    float maxI;
    float maxScore = 0;
	for (float i = 0; i < 1; i+=10.1)
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

    maxI = maxA;
    rightB = (maxB - 100) / 128;
    leftB = (maxB - 100) / 128;
    cerr << maxB << "\t" << rightB << endl;

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
