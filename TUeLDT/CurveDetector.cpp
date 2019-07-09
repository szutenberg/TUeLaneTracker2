/*
 * CurveDetector.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#include "CurveDetector.h"
#include "opencv2/opencv.hpp"
#include "BirdView.h"
#include "NeuralNetwork.h"

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
	lastConfidence = 0;
	lastAl = lastAr;
	lastBl = lastBr = 0;
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
	Sobel( input, mGradX, CV_32F, 1, 0, 7, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
	Sobel( input, mGradY, CV_32F, 0, 1, 7, 1, 0, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

	for (int i = 1; i < segmentsToRemove.size(); i++)
	{
		line(mGradY, segmentsToRemove[i-1], segmentsToRemove[i], cv::Scalar(0), 8);
		line(mGradX, segmentsToRemove[i-1], segmentsToRemove[i], cv::Scalar(0), 8);
	}

    mMask = (mGradX == 0);
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

	addWeighted(mProbMap_GradMag, 1.0, mProbMap_Gray, 0.0, 0, mProbVal, CV_8U);

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
	tmp.push_back(baseL);
	tmp.push_back(baseR);

	mBird.convertPointsToBird(tmp, tmp2);

	assert(tmp2.size() == 10);
    pxPerCm = ((double)(tmp2[9].x - tmp2[8].x))/mLaneFilter->LANE.AVG_WIDTH;

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


vector<int> CurveDetector::curve(double a, double b, double c)
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



void CurveDetector::calculateHistograms(cv::Mat img, cv::Mat ang, double* vals)
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
			//if (s == 0) val = 1;
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

		cout << s << " " << total << endl;
	}
}


void CurveDetector::plotHistograms(const char * name, double* vals)
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


void CurveDetector::filterHistorgrams(double* in, double *out)
{
	for (int s = 0; s < STRIPES_AMOUNT; s++)
	{
		for (int i = RANGE_FROM; i < RANGE_TO; i++) out[s * RANGE_N + i-RANGE_FROM] = 0;

		double maxV = 0.0000001;

		for (int i = RANGE_FROM + 4; i < RANGE_TO-4; i++)
		{
			double val = 0;
			for (int j = RANGE_FROM; j < RANGE_TO; j++)
			{
				val += in[s * RANGE_N + j - RANGE_FROM] * (1.0 - (double)abs(i-j)/((double)abs(i-j) + 0.3));
			}

			val = pow(val, 1.2 - (STRIPES_AMOUNT - s)*0.2 );
			out[s * RANGE_N + i - RANGE_FROM] = val;
			if (maxV < val) maxV = val;
		}

		for (int i = RANGE_FROM; i < RANGE_TO; i++)
		{
			out[s * RANGE_N + i - RANGE_FROM] /= maxV;
		}
	}
}

void CurveDetector::plotArray(const char * name, double *in, int N)
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


double CurveDetector::calcHistProb(double *in, double a, double b)
{
	double prob = 1;

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
	}

	return prob;
}


void CurveDetector::findCandidates(double *in, double *params, int* amt)
{
	*amt = 0;
	double arg[2000];
	double t[2000];
	double t2[2000];
	double t3[2000];
	double t4[2000];
	double probA[2000];
	double lastB = (lastBr + lastBl) / 2.0;


	int N = 0;
	for (double a = -0.1; a <= 0.1; a+=0.0005)
	{
		double maxProb = 0;

		for (double b = -0.5; b < 0.5; b+=0.01)
		{
			double prob = calcHistProb(in, a, b);
			double arg = (b - lastB)/0.04;
			arg*=arg;
			//prob *= exp(-(arg));


			if (maxProb < prob) maxProb = prob;
			//if (abs(lastAr/STRIP_H - a)<0.0001)  cerr << b << "\t" << prob << "\n";

		}
		t2[N] = 0;
		t3[N] = 0;
		arg[N] = a;

		double arg = 50.0;
		arg *= (a - lastAr*STRIP_H) * lastConfidence;
		probA[N] = exp(-arg*arg);
		t[N] = maxProb;
		N++;
	}

	plotArray("a", t, N);
	plotArray("probA", probA, N);

	N = 0;
	for(double a = -0.1; a <= 0.1; a+=0.0005)
	{
		t[N] *=  probA[N];
		N++;
	}

//	plotArray("a prob", t, N);

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

//	plotArray("t2", t2, N);

	maxVal = 0;
	for (int i = 0; i < N; i++)
	{
		double d = abs(i - candidate1);
		double val = t2[i] * (d / (d + 30.0));
		t3[i] = val;

		if (maxVal < val) maxVal = val, candidate2 = i;
	}

	t3[candidate2] *= -1;



//plotArray("t3", t3, N);
	t[candidate1] *= -1;
	t[candidate2] *= -1;


//	cerr << "\t" << arg[candidate1] << "\t" << arg[candidate2] << endl;

	plotArray("selected", t, N);


	double candidates[2] = {arg[candidate1], arg[candidate2]};

	N = 0;
	double canB[2];

	for (int cand = 0; cand < 1; cand++)
	{
		double a = candidates[cand];

		double maxP = 0;
		double maxB = 0;
		for (double b = -0.5; b < 0.5; b+=0.005)
		{
			double prob = calcHistProb(in, a, b);
			double arg = (b - lastB)/0.04;
			arg*=arg;
			prob *= exp(-(arg));

			t4[N++] = prob;
			if (maxP < prob) maxP=prob, maxB=b;
		}
		canB[cand] = maxB;
	}



//	plotArray("b", t4, N);
	params[0] = arg[candidate1] / STRIP_H;
	params[1] = canB[0];
	params[2] = arg[candidate2] / STRIP_H;
	params[3] = canB[1];
	*amt = 1;
}



void CurveDetector::calculateMatch(double *dest, int N, double a, double b, Mat img, Mat ang)
{
	Mat mask, imgRoi, angRoi, difRoi, tmp, imgRoi2, difRoi2;
	Mat img2;

	img.copyTo(img2);
	cv::Rect lROI;

	for (int i = 0; i < N; i++) dest[i] = 0;

	int STR_N = 10;

	int pos[STR_N+1], posC[STR_N+1];

	for (int i = 0; i < STR_N; i++)
	{
		pos[i] = BIRD_HEIGHT - BIRD_HEIGHT/2 * (double)sqrt(i) / sqrt(STR_N);

	}

	pos[STR_N] = BIRD_HEIGHT/2;

	for (int i = 0; i < STR_N; i++)
	{
		posC[i] = pos[i] - (pos[i] - pos[i+1]) * sqrt(0.5);
	}

	for (int s = 0; s < STR_N; s++)
	{
		lROI = cv::Rect(0, pos[s+1], img.cols, pos[s]-pos[s+1]);

		img(lROI).copyTo(imgRoi);
		ang(lROI).copyTo(angRoi);

		double y = BIRD_HEIGHT - posC[s];
		double dif = a * y * y + b * y;
		double angle = (2 * a * y + b) * 128;

		angRoi.copyTo(difRoi);
		difRoi -= angle;
		difRoi = abs(difRoi);
		tmp = difRoi + 40;
		difRoi = tmp - difRoi;

		divide(difRoi, tmp, difRoi2, 1, CV_32F);
		multiply(difRoi2, imgRoi, imgRoi2, 1, CV_8U);

		Mat dst;
		//if (s == 0) imshow("roi", imgRoi2);
		reduce(imgRoi2, dst, 0, CV_REDUCE_SUM, CV_32S);


		///cerr << s << ": " << pos[s] << "\t" << dif<<endl;
		int * ptr = dst.ptr<int>(0);
		for (int i = 0; i < dst.cols; i++)
		{
			int index = i - dif;
			if ((index >= 0) && (index < N)) dest[index] += ptr[i];
		}
	}
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


void CurveDetector::applyConfidenceGradient(cv::Mat& img)
{
	Mat plotL = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat plotR = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat mask;

	 vector<int> left  = curve(lastAl, lastBl, lastCl);
	 vector<int> right  = curve(lastAr, lastBr, lastCr);

	int width[5] = {150, 100, 75, 50, 25};
	int level[5] = {50, 75, 120, 200, 255};

	for (int j = 0; j < 5; j++)
	{
		for (int i = 50; i < left.size(); i+=50)
		{
			int scale = 1;
			if (i <= 300) scale = 8;
			if (i <= 600) scale = 3;

			line(plotL, Point(left[i-50], i-50), Point(left[i], i), CvScalar(level[j]), width[j] * scale);
			line(plotR, Point(right[i-50], i-50), Point(right[i], i), CvScalar(level[j]), width[j] * scale);
		}
	}

	mask = plotR > plotL;
	plotR.copyTo(plotL, mask);




	multiply(img, plotL, img, 1/256.0);
}

void normalize(double *c, int N)
{
	double maxV = 0.000001;
	for (int i = 0; i < N; i++)
		if (maxV < c[i]) maxV = c[i];
	for (int i = 0; i < N; i++)
		c[i] /= maxV;
}


void CurveDetector::trackEgoLane(double *c, int N)
{
	double c2[N];
	double sumC[N];

	c2[0] = c2[1] = c2[N-2] = c2[N-1] = 0;
	for (int i = 2; i < N-2; i++) c2[i] = c[i-2] + c[i+2] + c[i-1] * 2.0 + c[i+1] * 2.0 + c[i] * 4.0;

	normalize(c2, N);
	plotArray("c2", &c2[0], N);

	sumC[0]= 0;
	for (int i = 1; i < N; i++) sumC[i] = c2[i] + sumC[i-1];


	double maxP = 0.00001;
	int maxRd, maxLd;

	// can be optimized (only consider few peaks)
	for (int ld = -20; ld <= 20; ld++)
	{
		for (int rd = -20; rd <= 20; rd++)
		{
			int rc = lastCr + rd;
			int lc = lastCl + ld;

			double laneWidthDiff = rd-ld;
			double lanePosChange = (rd + ld)/2.0;
			double prob = exp(-laneWidthDiff * laneWidthDiff / 100.0);
			prob *= exp(-lanePosChange * lanePosChange / 100.0);

			prob *= c2[rc] * c2[lc];

			if (maxP < prob) maxP = prob, maxLd = ld, maxRd = rd;
		}
	}

	int lc = lastCl + maxLd;
	int rc = lastCr + maxRd;
	double snr = (c2[rc] + c2[lc]) / (sumC[rc - 5] - sumC[lc + 5]);

	cout << maxP << "\t" << snr << "\t"<< maxRd << "\t" << maxLd << "\n";


	lastCr += maxRd;
	lastCl += maxLd;



}



void CurveDetector::evaluateC(double *c, int N)
{
	double laneProbability[N];

	for (int i = 0; i < N; i++)
	{
		double widthCm = i / pxPerCm;
		double mi = mCfg->lane_avg_width;
		double var = mCfg->lane_std_width * mCfg->lane_std_width;
		laneProbability[i] = 0;
		if (widthCm >= mCfg->lane_min_width && widthCm <= mCfg->lane_max_width )
			//laneProbability[i] = exp( - (widthCm - mi)*(widthCm - mi) / (2.0 * var));
			laneProbability[i] = 1.0 - abs(widthCm - mi) * 0.001;
	}

	plotArray("c", &c[0], N);


	double tab[3*N];
	double tmp[N], tmp2[N];

	double maxV = 0.00001;
	int left1, left2;
	maxV = 0;
	for (int i = 0; i < N/2; i++)
	{
		if (maxV < c[i]) maxV = c[i], left1 = i;
	}

	maxV = 0;
	for (int i = 0; i < N; i++)
	{
		tmp2[i] = c[i];
		if (maxV < tmp2[i]) maxV = tmp2[i];
	}

	for (int i = 0; i < N; i++) tmp2[i] /= maxV;

	tmp2[left1] *= -1;

	maxV = 0;
	for (int i = 0; i < N/2; i++)
	{
		double d = abs(i - left1);
		double val = tmp2[i] * (d / (d + 30.0));
		tmp[i] = val;

		if (maxV < val) maxV = val, left2 = i;
	}

	c[left2] *= -1;
	plotArray("cSel", &tmp2[0], N);

	for (int i = 0; i < N; i++)
	{
		tab[i] = 0;
		tab[i+N] = 0;
		tab[i+N+N] = 0;

		if (i > left1) tab[i] = tmp2[i] * laneProbability[i-left1];
		if (i > left2) tab[N+i] = tmp2[i] * laneProbability[i-left2];
	}

	maxV = 0;
	for (int i = 0; i < 3*N; i++)
	{
		if (maxV < tab[i]) maxV = tab[i];
	}

	for (int i = 0; i < 3*N; i++)
	{
		tab[i] /= maxV;
	}

	tab[left1] = -1;
	tab[N + left2] = -1;

	plotArray("tab", &tab[0], N*2);

	int left[2];
	left[0] = left1;
	left[1] = left2;

	int right[2];
	double conf[2];
	for (int s = 0; s < 2; s++)
	{
		double maxV = 0;
		int offset = s * N;
		for (int i = left[s]; i < N; i++)
		{
			if (maxV < tab[i + offset]) maxV = tab[i + offset], right[s] = i;
		}

		double sum = 0;
		for (int i = left[s]+1; i < right[s] - 20; i++)
		{
			sum += tab[i + offset];
		}

		double laneProb = 0.999999 - sum/tab[right[s] + offset]/15.0;
		if (laneProb < 0.1) laneProb = 0.1;
		double probability = laneProb;

		double leftProb = exp(-(left[s]-lastCl)*(left[s]-lastCl)/400.0)  + 0.5;
		leftProb /= 1.5;

		double rightProb = exp(-(right[s]-lastCr)*(right[s]-lastCr)/400.0)  + 0.5;
		rightProb /= 1.5;


		conf[s] = leftProb * rightProb * probability;
		cerr << s << " : " << leftProb*rightProb << "\t" << probability << "\t" << lastCl << "->" << left[s]  << " \t" << lastCr << "->"<< right[s] << "\n";

		//cerr << leftProb << "\t" << rightProb << "\n";
		//cerr << leftProb * rightProb * probability << endl;

	}
	int s = 1;
	if (conf[0] > conf[1]) s = 0;

	lastCl = left[s];
	lastCr = right[s];
	lastConfidence = conf[s];
	cerr << s << " " << conf[s] << endl;

	//plotArray("laneProbability", &laneProbability[0], N);
}


Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return img;
}


void CurveDetector::filterLaneMarkings(cv::Mat img, cv::Mat& laneMarkingsVal)
{
	Mat tmp, tmp2;

	GaussianBlur(img, tmp, cv::Size(29, 29), 10, 15);
	GaussianBlur(img, img, cv::Size(3, 3), 0.5, 0.5);

	addWeighted(tmp, -19.5, img, 18.0, -50.0, tmp2);
	for (int i = 1; i < segmentsToRemove.size(); i++)
	{
		line(tmp2, segmentsToRemove[i-1], segmentsToRemove[i], cv::Scalar(0), 24);
	}

	//imshow("blur", tmp);


	img -= 128;


	multiply(img, tmp2, tmp, 1/128.0);
	imshow("tmp2", tmp);

	tmp.copyTo(laneMarkingsVal);
/*
	cv::Mat map = Mat::zeros(img.rows, img.cols, CV_8U);

	cv::Mat binaryImage = tmp2 > 30;





	Mat mask = map == 255;
	map.setTo(255, binaryImage);
	imshow("map", map);

	GaussianBlur(binaryImage, binaryImage, cv::Size(13, 13), 5, 5);
	//resize(binaryImage, laneMarkingsVal, cv::Size(), 0.7, 0.7, INTER_CUBIC);
	imshow("bin", binaryImage);

	binaryImage.copyTo(laneMarkingsVal);
*/

/*



	Mat heh1, heh2, heh3, heh4;
	GaussianBlur(img, heh1, cv::Size(49, 49), 50, 50);
	GaussianBlur(img, heh2, cv::Size(29, 29), 20, 20);

	addWeighted(heh1, 1, heh2, -1, 0.0, heh3);
	Mat mask2;

	mask2 = heh3 > 1;
	img.copyTo(heh4, mask2);
	heh4.setTo(255, mask2);
	GaussianBlur(heh4, heh4, cv::Size(21, 21), 10, 10);
	mask2 = heh4 > 100;

	tmp.setTo(0, mask2);
	imshow("tmp", tmp);

	imshow("heh3", heh4);

	//return 1;
*/
}


int rr = 0;
int CurveDetector::run(cv::UMat& frame, LaneModel* Lane)
{

	if (rr++ < 7)
	{
		lastCl = birdLeftStart.x;
		lastCr = birdRightStart.x;
		lastAl = lastAr = 0;
		lastBr = lastBl = 0;
	}

	Mat birdRaw, tmp;

	Point mO_ICCS_ICS(frame.cols/2, frame.rows/2);
	mInput = frame;

	UMat nnRoi;
	Mat nnOut = NeuralNetwork::getResult();
	if (nnOut.rows)
	{
		imshow("nnOut", nnOut);
		prepareROI(nnOut.getUMat(ACCESS_READ), nnRoi);
		imshow("nnROI", nnRoi);
	}


	prepareROI(mInput, mROI);





	GaussianBlur(mROI, mROIblurred, cv::Size(3,3), 3, 3);

	//if (mCfg->display_graphics) imshow("mROIblurred", mROIblurred);

	tmp = mROIblurred.getMat(ACCESS_READ);

	setParams(Lane, tmp);

	birdRaw = mBird.applyTransformation(tmp);

	if (mCfg->display_graphics) imshow("birdRaw", birdRaw);
	Mat laneMarkings;
	filterLaneMarkings(birdRaw, laneMarkings);
	Mat laneMarkingsBlurred;
	GaussianBlur(laneMarkings, laneMarkingsBlurred, cv::Size(5, 5), 3, 3);

	//imshow("blurred", laneMarkingsBlurred);


	GaussianBlur(birdRaw, birdRaw, cv::Size(13, 13), 5, 5);

	computeMap(birdRaw, mFrMag, mFrAng);

	mFrMag /= 256;
	Mat markMag, markAng;
	computeMap(laneMarkingsBlurred, markMag, markAng);

	if (nnOut.rows)
	{
		Mat nnBird = mBird.applyTransformation(nnRoi.getMat(ACCESS_READ));
		imshow("nnBird", nnBird);
		nnBird.convertTo(nnBird, CV_16S);
		multiply(markMag, nnBird, markMag, 1.0/255, CV_8U);

	}

	Mat mask = markMag < 1;
	laneMarkings.setTo(0, mask);
	mask = laneMarkings > mFrMag;
	laneMarkings.copyTo(mFrMag, mask);
	markAng.copyTo(mFrAng, mask);




	bufferFiltering(mCfg->buffer_count);


	//applyConfidenceGradient(mFrMag);
	if (mCfg->display_graphics) imshow("mFrMag", mFrMag);

	Mat debugFrame;
	mFrMag.copyTo(debugFrame);
    cvtColor(debugFrame, debugFrame, COLOR_GRAY2BGR);

    line(debugFrame, Point(birdLeftStart + Point2f(2, 2)), Point(birdLeftStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdLeftStart + Point2f(-2, 2)), Point(birdLeftStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);

    line(debugFrame, Point(birdRightStart + Point2f(2, 2)), Point(birdRightStart + Point2f(-2, -2)), CvScalar(255, 0, 0), 2);
    line(debugFrame, Point(birdRightStart + Point2f(-2, 2)), Point(birdRightStart + Point2f(2, -2)), CvScalar(255, 0, 0), 2);

    double hist[STRIPES_AMOUNT*RANGE_N], hist2[STRIPES_AMOUNT*RANGE_N];
    double hist3[STRIPES_AMOUNT*RANGE_N];
    calculateHistograms(mFrMag, mFrAng, &hist[0]);
    plotHistograms("before Filtering", &hist[0]);
    filterHistorgrams(&hist[0], &hist2[0]);
    plotHistograms("after Filtering", &hist2[0]);


    double params[10];
    int candidatesN;
    findCandidates(&hist2[0], &params[0], &candidatesN);

    double cHists[10000];
    calculateMatch(&cHists[0], mFrMag.cols, params[0], params[1], mFrMag, mFrAng);
    calculateMatch(&cHists[mFrMag.cols], mFrMag.cols, params[2], params[3], mFrMag, mFrAng);

    int markerWidth = mCfg->lane_marker_width * pxPerCm;
    markerWidth = 10; // fixMe

    for (int n = 0; n < candidatesN; n++)
    {
    	int offset = n * mFrMag.cols;

    	for (int i = 0; i < mFrMag.cols; i++)
    	{
    		hist3[offset + i] = 0;
    	}

    	for (int i = 0; i < mFrMag.cols; i++)
    	{
    		for (int j = -markerWidth/2; j <= markerWidth/2; j++)
    		{
    			if (i+j < 0) continue;
    			if (i+j >= mFrMag.cols) continue;
    			hist3[offset + i] += cHists[offset + i + j];
    		}
    	}
    }

    //evaluateC(&cHists[0], mFrMag.cols);
    trackEgoLane(&cHists[0], mFrMag.cols);


    double maxV = 0;
    for (int i = 0; i < mFrMag.cols*candidatesN; i++) if (maxV < cHists[i]) maxV = cHists[i];
    for (int i = 0; i < mFrMag.cols*candidatesN; i++) cHists[i] /= maxV;

	//plotArray("c", &cHists[0], mFrMag.cols*candidatesN);


	double leftSidePeak = 0;
	double rightSidePeak = 0;
	double between = 0;

	for (int i = -10; i < 10; i++)
	{
		leftSidePeak += cHists[i + (int)birdLeftStart.x];
		rightSidePeak += cHists[i + (int)birdRightStart.x];
	}

	for (int i = birdLeftStart.x + 10; i <= birdRightStart.x - 10; i++) between += cHists[i];


	lastConfidence =  (leftSidePeak + rightSidePeak)/(between + leftSidePeak + rightSidePeak);


	cerr << lastConfidence << endl;



	maxV = 0;
	for (int i = 0; i < mFrMag.cols*candidatesN; i++) if (maxV < hist3[i]) maxV = hist3[i];
	for (int i = 0; i < mFrMag.cols*candidatesN; i++) hist3[i] /= maxV;

//	plotArray("hist3 c", &hist3[0], mFrMag.cols*candidatesN);





	lastAl = lastAr = params[0];
	lastBl = lastBr = params[1];



    rightB = (maxB - 100) / 128;
    leftB = (maxB - 100) / 128;
    vector<int> test;


	test  = curve(lastAl, lastBl, lastCl);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}
	Lane->curveL.clear();

	vector<Point2f> out;
	test  = curve(lastAl, lastBl, lastCl);
	for (int i = test.size() - 1; i >= 0; i-=50)
	{
		out.push_back(Point2f(test[i], i));
	}

	mBird.invertPoints(out, Lane->curveL);
	out.clear();

	test  = curve(lastAr, lastBr, lastCr);
	for (int i = 1; i < test.size(); i++)
	{
		line(debugFrame, Point(test[i-1], i-1), Point(test[i], i), CvScalar(0,0, 255), 2);
	}

	test  = curve(lastAr, lastBr, lastCr);
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
