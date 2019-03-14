/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector3.h"
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

const LineSegment NOT_DETECTED = {Point2f(-1, -1), Point2f(-1, -1), -1, -1};
const float RAD_TO_DEG = 180.0 / 3.141592653;

#define DEBUG_CD

// Source
void polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order)
{
	Eigen::MatrixXd A(xv.size(), order+1);
	Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
	Eigen::VectorXd result;

	assert(xv.size() == yv.size());
	assert(xv.size() >= order+1);

	// create matrix
	for (size_t i = 0; i < xv.size(); i++)
	for (size_t j = 0; j < order+1; j++)
		A(i, j) = pow(xv.at(i), j);

	// solve for linear least squares fit
	result = A.householderQr().solve(yv_mapped);

	coeff.resize(order+1);
	for (size_t i = 0; i < order+1; i++)
		coeff[i] = result[i];
}



void fitPointsYX(std::vector<Point2f> &points, std::vector<Point2f> &curve, Point zero=Point(0,0))
{
	std::vector<double> xv;
	std::vector<double> yv;
	std::vector<double> coeff;

	for (Point2f p : points)
	{
		xv.push_back(zero.y - p.y );
		yv.push_back(p.x - zero.x);
	}
	polyfit(xv, yv, coeff, 2);

	for (int y = 695; y >= 0; y-=50)
	{
		float yc = zero.y - y ;
		float xc = yc * yc * coeff[2] + yc * coeff[1] + coeff[0] + zero.x;
		curve.push_back(Point2f(xc, y));
	}
	//printf("%lf\t%lf\t%lf\n", coeff[2], coeff[1], coeff[0]);

}

int CurveDetector3::detectCurve(const cv::Mat& img, Point start, std::vector<Point2f> &curve)
{
#ifdef DEBUG_CD
	Mat debugBird;
    img.copyTo(debugBird);
    cvtColor(debugBird, debugBird, COLOR_GRAY2BGR);
#endif // DEBUG_CD

    float totalScore = 0;

	LineSegment best = NOT_DETECTED;
	float maxD = 0.1;

	for (LineSegment s : *seg)
	{
		if (abs(s.a.x - start.x) > 12) continue;
		if (abs(s.angle) > 10) continue;

		Point2f dst = s.a - Point2f(start);
		float dif = sqrt(dst.x * dst.x + dst.y * dst.y);
		if (dif < 50) dif = 50;
		float d = s.score * s.score / dif;

		if (maxD < d)
		{
			maxD = d;
			best = s;
		}
	}

	if (best == NOT_DETECTED) return -1;
	vector<LineSegment> detSeg;

#ifdef DEBUG_CD
	for (LineSegment s : *seg)
	{
		line(debugBird, s.a, s.b, CvScalar(0, s.score*2.5, 0), 2);
	}
	cv::Point2f c1(-3, 0);
	cv::Point2f c2(3, 0);
	cv::Point2f c3(0, 3);
	cv::Point2f c4(0, -3);
#endif // DEBUG_CD

	do
	{
#ifdef DEBUG_CD
		line(debugBird, best.a + c1, best.a + c2, CvScalar(0, 0, 200), 2);
		line(debugBird, best.a + c3, best.a + c4, CvScalar(0, 0, 200), 2);

		line(debugBird, best.b + c1, best.b + c2, CvScalar(0, 0, 200), 2);
		line(debugBird, best.b + c3, best.b + c4, CvScalar(0, 0, 200), 2);
#endif // DEBUG_CD

		LineSegment last = best;
		Point2f lastVec = last.b - last.a;
		float lastLen = sqrt(lastVec.x * lastVec.x + lastVec.y * lastVec.y);
		totalScore += lastLen * last.score;
		detSeg.push_back(best);


		float maxD = 0.1;
		best = NOT_DETECTED;

		for (LineSegment s : *seg)
		{
			if (last.b.y < s.a.y) continue;
			if (abs(s.angle - last.angle) > 12) continue;

			float angleDirNext; // angle between direction of last and beginning of s

			Point2f vecD = s.a - last.b;
			angleDirNext = (atan(vecD.x / vecD.y) - atan(lastVec.x / lastVec.y)) * RAD_TO_DEG;
			if (abs(angleDirNext) > 7) continue;

			float squaredDist = (vecD.x * vecD.x + vecD.y * vecD.y);
			if (squaredDist < 50*50) squaredDist = 50*50;

			float d = s.score * s.score / squaredDist;

			if (maxD < d)
			{
				maxD = d;
				best = s;
			}
		}
	}while(best != NOT_DETECTED);



	for (LineSegment seg : detSeg)
	{
		Point2f vec = seg.b - seg.a;
		float len = abs(vec.y);
		vec /= len;

		float div = 1000000;

		// a^2 - b^2 = S * div/score
		// b = sqrt(a^2 - S * div/score)

		Point2f pt = seg.a;
		int i = 1;

		while(pt.y > seg.b.y)
		{
			curve.push_back(pt);
			float ydif = seg.a.y - sqrt(seg.a.y * seg.a.y - i * div / seg.score);
			pt = seg.a + vec * ydif;
			i++;
		}

		curve.push_back(seg.b);

	}


	for (Point2f p : curve)
	{
		line(debugBird, p + c1, p + c2, CvScalar(200, 0, 0), 2);
		line(debugBird, p + c3, p + c4, CvScalar(200, 0, 0), 2);
	}


	if (curve.size() > 3)
	{
		std::vector<Point2f> newCurve;
		fitPointsYX(curve, newCurve, start);
		curve = newCurve;
	}


#ifdef DEBUG_CD
	imshow(name.c_str(), debugBird);
#endif  // DEBUG_CD

	return totalScore/100;
}

