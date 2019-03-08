/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector3.h"

using namespace std;

int CurveDetector3::detectLane(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &lane)
{
	int maxVal = -1;
	Point2f vec = p2 - p1;
	float len = vec.y / (-200.0);
	p2 = p1 + Point(vec / len);

	for (int xoffset = -60; xoffset <= 60; xoffset +=20)
	{
		std::vector<Point> tmpLane;
		Point pt1(p1.x + xoffset, p1.y);
		Point pt2(p2.x + xoffset, p2.y);

		int value = adjustLane(img, pt1, pt2, tmpLane);
		if (value > maxVal)
		{
			maxVal = value;
			lane = tmpLane;
		}
	}
	return maxVal;
}


int CurveDetector3::adjustLane(const cv::UMat& img, Point p1, Point p2, std::vector<Point> &lane)
{
	int maxVal = -1;
	Point a, b;
	float div;
	std::vector<Point> points;
	points = selectNextPoints(img, p1, Point2f(p2 - p1), 1);

	for (size_t i = 0; i < points.size(); i++)
	{
		for (size_t j = i+1; j < points.size(); j++)
		{
			Point pc1 = points[i];
			Point pc2 = points[j];

			Point2f vec = pc2 - pc1;
			if (abs(vec.y) < 5) continue;

			//cout << p1 << p2 << pc1 << pc2 << endl;

			if (p1.y != pc1.y)
			{
				div = vec.y / (p1.y - pc1.y);
				vec /= div;
				pc1 += Point(vec);
			}

			if (p2.y != pc2.y)
			{
				div = vec.y / (p2.y - pc2.y);
				vec /= div;
				pc2 += Point(vec);
			}

			Point2f vecA(p2-p1);
			Point2f vecB(pc2-pc1);

			float degA = atan(vecA.y / vecA.x) * 180.0 / 3.14;
			float degB = atan(vecB.y / vecB.x) * 180.0 / 3.14;
			//cout << degA << " " << degB << endl;
			float dif = abs(degA - degB);
			if (dif > 20) continue;

			//cout << p1 << p2 << pc1 << pc2 << endl;

			int value = calcScore(img, pc1, pc2, 1);
			//cout << "value = " << value << endl;
			if (value > maxVal)
			{
				maxVal = value;
				a = pc1;
				b = pc2;
			}
		}
	}
	lane.push_back(a);
	lane.push_back(b);

	return maxVal;
}

// not used
void CurveDetector3::grabPoints(Point a, Point b, std::vector<Point> &points)
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


inline int CurveDetector3::isPointOutOfRange(Point a, int width, int height)
{
	return ((a.x < 1) || (a.y < 1) || (a.x > (width - 1)) || (a.y > (height - 1)));
}


std::vector<Point> CurveDetector3::selectNextPoints(const cv::UMat& img, Point pt, Point2f vec, int step)
{
	std::vector<Point> res;
	vector<int> resValues;

	float vecLen = sqrt(vec.x * vec.x + vec.y * vec.y);
	vec /= vecLen;

	Point2f vecPerp = vec;
	swap(vecPerp.x, vecPerp.y);
	vecPerp.x *= -1.0;
	vecPerp *= 30.0;

	for (int i = 0; i < 15; i++)
	{
		Point2f c(pt);
		float len = 50.0 + i * 20.0; // TODO - remove magic numbers

		c.x += vec.x * len;
		c.y += vec.y * len;

		Point e1(c + vecPerp);
		Point e2(c - vecPerp);

		if (isPointOutOfRange(e1, img.cols, img.rows) ||
				isPointOutOfRange(e2, img.cols, img.rows))
		{
			continue;
		}

/*
		vector<cv::Point> mark;
		mark.push_back(e1);
		mark.push_back(e2);
		debugCurves.push_back(mark);
*/


		Point maxPos(0, 0);

		Point a = e1;
		Point b = e2;

		float slope = (float)(b.y - a.y) / (b.x - a.x);
		int dirx = 0;
		int diry = 0;

		if (abs(slope) < 1)
		{
			dirx = 1;
			if (a.x > b.x) swap(a, b);
		}
		else
		{
			diry = 1;
			if (a.y > b.y) swap(a, b);
			slope = 1 / slope;
		}

		int from = dirx * a.x + diry * a.y;
		int to   = dirx * b.x + diry * b.y;

		int counter = 0;

		const int WIDTH = 33;
		int maxVal = WIDTH * 10;

		Point positions[WIDTH];
		int values[WIDTH];
		for (int i = 0; i < WIDTH; i++) values[i] = 0;

		int avg = 0;
		int avgIt = 0;

		float p = diry * a.x + dirx * a.y;

		float range = 0.5 * sqrt(slope * slope + 1);

		for (int i = from; i <= to; i++, p+=slope)
		{
			for (int j = p - range ; j <= p + range ; j++, counter++)
			{
				// TODO: optimize - make two loops (dirx=0 and dirx=1)
				Point pos(dirx * i + diry * j, dirx * j + diry * i);

				if (isPointOutOfRange(pos, img.cols, img.rows))
				{
					counter--;
					continue;
				}

				int val = img.getMat(ACCESS_READ).at<uchar>(pos);

				avg += val;
				avg -= values[avgIt];

				if (avg > maxVal)
				{
					maxVal = avg;
					maxPos = positions[(avgIt + WIDTH/2)%WIDTH];
				}

				values[avgIt] = val;
				positions[avgIt] = pos;
				avgIt++;
				avgIt %= WIDTH;
			}
		}

		if (maxPos.x)
		{
			res.push_back(maxPos);
			resValues.push_back(maxVal);
		}
	}

	/*cout << "selected " << res.size() << endl;
	for (size_t i = 0; i < res.size(); i++)
	{
		cout << i << " :  " << res[i] << "\t" << resValues[i] << endl;
	}
*/
	std::vector<Point> finalRes;

	if (res.size() <= 8) return res;

	while(finalRes.size() != 8)
	{
		Point pt;
		int maxi;
		int maxVal = 0;

		for (size_t i = 0; i < res.size(); i++)
		{
			if (resValues[i] > maxVal)
			{
				maxi = i;
				maxVal = resValues[i];
			}
		}

		finalRes.push_back(res[maxi]);
		resValues[maxi] = -1;
	//	cout << "Putting " << maxVal << endl;
	}

	return finalRes;
}

int CurveDetector3::calcScore(const cv::UMat& img, Point a, Point b, float d)
{
	int res = 0;
	if (a == b) return 0;
	float slope = (float)(b.y - a.y) / (b.x - a.x);
	int dirx = 0;
	int diry = 0;

	if (abs(slope) < 1)
	{
		dirx = 1;
		if (a.x > b.x) swap(a, b);
	}
	else
	{
		diry = 1;
		if (a.y > b.y) swap(a, b);
		slope = 1 / slope;
	}

	int from = dirx * a.x + diry * a.y;
	int to   = dirx * b.x + diry * b.y;

	if (from < 0) from = 0;
	if ((dirx) && (to >= img.cols)) to = img.cols - 1;
	if ((diry) && (to >= img.rows)) to = img.rows - 1;

	int counter = 0;

	float p = diry * a.x + dirx * a.y;

	float range = d * sqrt(slope * slope + 1);

	for (int i = from; i <= to; i++, p+=slope)
	{
		for (int j = p - range; j <= p + range; j++, counter++)
		{
			// TODO: optimize - make two loops (dirx=0 and dirx=1)
			Point pos(dirx * i + diry * j, dirx * j + diry * i);
			if (isPointOutOfRange(pos, img.cols, img.rows)){
				counter--;
				continue;
			}
			res += img.getMat(ACCESS_READ).at<uchar>(pos);
		}
	}

	return res / counter;
}

