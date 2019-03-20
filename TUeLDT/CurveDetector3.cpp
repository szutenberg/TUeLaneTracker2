/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector3.h"
#include "polyFit.h"
using namespace std;

const LineSegment NOT_DETECTED = {Point2f(-1, -1), Point2f(-1, -1), -1, -1};
const float RAD_TO_DEG = 180.0 / 3.141592653;

extern int debugX;

#define DEBUG_CD

std::vector<double> fit(std::vector<Point2f>& points, Point2f zero)
{
	std::vector<double> xv;
	std::vector<double> yv;
	std::vector<double> coeff;

	for (Point2f p : points)
	{
		xv.push_back(zero.y - p.y );
		yv.push_back(p.x - zero.x);
	}
	polyFit(xv, yv, coeff, 1);

	return coeff;
}

float value(std::vector<double>& coeff, Point2f zero, float y)
{
	float yc = zero.y - y ;
	float xc = yc * yc * coeff[2] + yc * coeff[1] + coeff[0] + zero.x;
	return xc;
}


float angle(std::vector<double>& coeff, Point2f zero, float y)
{
	float yc = zero.y - y ;
	float xc = 2.0 * yc * coeff[2] + coeff[1];

	return atan(xc) * 180.0 / 3.14;
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
	polyFit(xv, yv, coeff, 2);

	for (int y = zero.y; y >= 1; y-=10)
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
/*
    curve.push_back(start - Point(0, 500));
    curve.push_back(start);
    curve.push_back(Point(start.x, 1));
*/





    float totalScore = 0;

	LineSegment best = NOT_DETECTED;
	float maxD = 0.1;

	for (LineSegment s : *seg)
	{
		if (abs(s.a.x - start.x) > 12) continue;
		if (abs(s.angle) > 10) continue;

		Point2f dst = s.a - Point2f(start);
		float dif = sqrt(dst.x * dst.x + dst.y * dst.y);
		if (dif < 10) dif = 10;
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
		line(debugBird, s.a, s.b, CvScalar(0, s.score, 0), 2);
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

			if ((s.score - last.score) < -30) continue;
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



	for (LineSegment s : detSeg)
	{/*
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
		}*/
		curve.push_back(s.a);
		curve.push_back(s.b);



		for (LineSegment seg2 : *seg)
		{
			if (seg2 == s) continue;
			if ((s.score - seg2.score) > 10) continue;
			Point2f v = seg2.a - s.a;
			float likehood = v.x * v.x + v.y * v.y;
			float angdif = abs(s.angle - seg2.angle);
			if (likehood > 26) continue;
			if (angdif > 2) continue;
			curve.push_back(seg2.a);
			curve.push_back(seg2.b);
			//cout << s << seg2 << "Likehood = " << likehood << " angle = " << angdif << "\n";
		}
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
	if (debugX == 0) imshow(name.c_str(), debugBird);
#endif  // DEBUG_CD

	return totalScore/100;
}



int CurveDetector3::detectCurve2(const cv::Mat& img, Point2f s1, Point2f s2, std::vector<Point2f> &curve)
{
#ifdef DEBUG_CD
	Mat debugBird;
    img.copyTo(debugBird);
    cvtColor(debugBird, debugBird, COLOR_GRAY2BGR);
#endif // DEBUG_CD
/*
    curve.push_back(start - Point(0, 500));
    curve.push_back(start);
    curve.push_back(Point(start.x, 1));
*/
    curve.clear();
    float totalScore = 0;

	float maxD = 0.1;

	Point2f v = s2 - s1;
	v /= 2.0;
	for (int i = 0; i < 2; i++)
	{
		Point2f t = s1;
		t += i * v;
		curve.push_back(t);
	}
	curve.push_back(s2);


#ifdef DEBUG_CD
	for (LineSegment s : *seg)
	{
		line(debugBird, s.a, s.b, CvScalar(0, s.score, 0), 2);
	}
	cv::Point2f c1(-3, 0);
	cv::Point2f c2(3, 0);
	cv::Point2f c3(0, 3);
	cv::Point2f c4(0, -3);
#endif // DEBUG_CD

	std::vector<Point2f> newCurve;
	//fitPointsYX(curve, newCurve, s1);

	for (Point2f p : newCurve)
	{
		line(debugBird, p + c1, p + c2, CvScalar(200, 0, 0), 2);
		line(debugBird, p + c3, p + c4, CvScalar(200, 0, 0), 2);
	}



	LineSegment best = NOT_DETECTED;

	float y = s1.y;

	do
	{
		float maxScore = 1;

		vector<double> coeff = fit(curve, s1);
		best = NOT_DETECTED;
		for (LineSegment s : *seg)
		{
			if (y < s.a.y) continue;

			float angleDif = abs(s.angle - angle(coeff, s1, s.a.y));
			float xDif = abs(s.a.x - value(coeff, s1, s.a.y));
			float yDif = y - s.a.y;
			if (yDif < 100) yDif = 101;


			float angleCoeff = 1.0 - angleDif / (angleDif + 5);
			float xCoeff = 1.0 - xDif / (xDif + 5);


			float score = angleCoeff * xCoeff * xCoeff * sqrt(s.score) / (yDif - 100);


			if (score > maxScore)
			{
				best = s;
				maxScore = score;
				cout << "Canidate" << name << score << "\t" << "\t" << s;
			}
		}

		if (best != NOT_DETECTED)
		{
			cout << best;
			Point2f v = best.b - best.a;
			float len = sqrt(v.x * v.x + v.y * v.y);
			float segments = (int)(len / 20);

			curve.push_back(best.a);
			v /= segments;
			for (int i = 1; i < segments; i++)
			{
				curve.push_back(best.a + i * v);
			}
			y = best.b.y;
			line(debugBird, best.a, best.b, CvScalar(0, 0, best.score*2.0), 2);
		}


	}while(best != NOT_DETECTED);




#ifdef DEBUG_CD
	if (debugX == 0) imshow(name.c_str(), debugBird);
#endif  // DEBUG_CD


	fitPointsYX(curve, newCurve, s1);
	curve.clear();
	curve = newCurve;



	return 1;
}
