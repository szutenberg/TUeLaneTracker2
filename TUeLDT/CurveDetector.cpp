/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector.h"

using namespace std;

int CurveDetector::detectCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &curve)
{
	cout << "detectCurve " << p1 << p2 << endl;

	int maxVal = -1;
	Point2f vec(p2 - p1);
	float len = sqrt(vec.x * vec.x + vec.y * vec.y);
	vec /= len;

	p1 = p2;

	for (int xoffset = -30 + left * 20; xoffset <= 30 + left * 20; xoffset +=10)
	{
		std::vector<Point> tmpCurve;
		Point start = p1;
		start.x += xoffset;

		Point pos;
		pos.x = start.x + vec.x * 80.0;
		pos.y = start.y + vec.y * 80.0;
		int value = computeCurve(img, start, pos, tmpCurve) * (50 + xoffset * left);
		if (value > maxVal)
		{
			maxVal = value;
			curve = tmpCurve;
		}
	}

	return maxVal;
}

int CurveDetector::computeCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &curve)
{
	cout << "computeCurve" << p1 << p2 << endl;
	curve.push_back(p1);
	curve.push_back(p2);
	int confidence = 0;
	MAX_STEPS_AMOUNT = 10; //TODO
	for (int step = 0; step < MAX_STEPS_AMOUNT; step++)
	{
		std::vector<Point> points;
		points = selectNextPoints(img, p2, Point2f(p2 - p1));

		int maxVal = 0;
		Point maxPos(0,0);

		for (Point pc1 : points)
		{
			std::vector<Point> points2;
			points2 = selectNextPoints(img, pc1, Point2f(pc1 - p2));

			for (Point pc2 : points2)
			{
				int value = calcScore(img, p2, pc1) + calcScore(img, pc1, pc2);
				if (value > maxVal)
				{
					maxVal = value;
					maxPos = pc1;
				}
			}
		}

		if (maxPos.y < 50)
		{
			return confidence;
		}

		Point vec(maxPos - p2);
		float len = sqrt(vec.x * vec.x + vec.y * vec.y);
		confidence += maxVal * len;

		curve.push_back(maxPos);
		p1 = p2;
		p2 = maxPos;
	}

	return confidence;
}


// TODO: optimize it
void CurveDetector::grabPoints(Point a, Point b, std::vector<Point> &points)
{
	Point e1(min(a.x, b.x), min(a.y, b.y));
	Point e2(max(a.x, b.x) + 1, max(a.y, b.y) + 1);

	// Ax + By + C = 0
	// d = |Ax + By + C| / sqrt(A*A + B*B)

	int lA = a.y - b.y;
	int lB = b.x - a.x;
	int lC = a.x * b.y - b.x * a.y;
	int div = lA * lA + lB * lB;

	for (int x = e1.x; x <= e2.x; x++)
	{
		for (int y = e1.y; y <= e2.y; y++)
		{
			int d = (lA * x + lB * y + lC);
			d *= d;

			if (d <= div)
			{
				points.push_back(Point(x, y));
			}
		}
	}
}


inline int CurveDetector::isPointOutOfRange(Point a, int width, int height)
{
	return ((a.x < 10) || (a.y < 10) || (a.x > (width - 10)) || (a.y > (height - 10)));
}


std::vector<Point> CurveDetector::selectNextPoints(const cv::UMat& img, Point a, Point2f vec)
{
	cout << "selectNextPoints" << a << vec << endl;
	std::vector<Point> res;

	float len = sqrt(vec.x * vec.x + vec.y * vec.y);
	vec /= len;

	Point2f vecPerp = vec * 2.0;

	for (int i = 0; i < 4; i++)
	{
		Point2f c(a);
		c.x += vec.x * (1<<i) * 10.0;
		c.y += vec.y * (1<<i) * 10.0;
		Point e1(c + vecPerp);
		Point e2(c - vecPerp);

		if (isPointOutOfRange(e1, img.cols, img.rows) ||
				isPointOutOfRange(e2, img.cols, img.rows))
		{
			continue;
		}

		int maxVal = 1;
		Point maxPos(0, 0);
		float curAng = atan(vec.y / vec.x);
		std::vector<Point> candidatePoints;
		grabPoints(e1, e2, candidatePoints);

		for (Point pos : candidatePoints)
		{
			int val = img.getMat(ACCESS_READ).at<uchar>(pos);
			std::cout << pos << " " << val << std::endl;

			if (val > maxVal)
			{
				Point2f newVec(pos - a);
				float newAng = atan(newVec.y / newVec.x);

				float dif = abs(newAng - curAng) * 180.0 / 3.14;
				if (dif > 10)
				{
					printf("selectNextPoints(i = %d): dif = %f\n", i, dif);
					continue;
				}

				maxVal = val;
				maxPos = pos;
			}
		}

		if (maxPos.x) res.push_back(maxPos);
	}

	// TODO - select only two
	return res;
}


int CurveDetector::calcScore(const cv::UMat& img, Point a, Point b)
{
	int res = 0;
	std::vector<Point> points;
	grabPoints(a, b, points);
	for (Point pos : points)
	{
		res += img.getMat(ACCESS_READ).at<uchar>(pos);
	}

	return res / points.size();
}

