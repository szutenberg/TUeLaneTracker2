/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector3.h"
#include <Eigen/QR>


using namespace std;
using namespace Eigen;


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
		if (p.y < 350) continue;
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

int CurveDetector3::detectCurve(const cv::UMat& img, Point p1, Point p2, std::vector<Point2f> &curve)
{
	int maxVal = -1;
	Point a, b;
	std::vector<Point> points;

	float STEP = 150;
	Point prev = p1;
	int maxSum = -1;
	curve.push_back(p1);
	float lastAngle = 0;
	int fromX = tan(lastAngle - 7.0 * 3.14 / 180.0) * STEP;
	int toX = tan(lastAngle + 7.0 * 3.14 / 180.0) * STEP;


	for (int i = 0; i < 100; i++)
	{
		if (prev.y <= 200) break;
		Point maxPoint(prev.x, prev.y - STEP);

		maxSum = 0;
		for (int x = prev.x + fromX; x <= prev.x + toX; x++)
		{
			Point trash;
			Point maxPointT = Point(x, prev.y - STEP);
			int value = calcScore(img, Point(prev.x, prev.y), Point(x, prev.y - STEP), 0, maxPointT);
			//int value1 = calcScore(img, Point(prev.x - 5, prev.y), Point(x - 5, prev.y - STEP), 1, trash);
			//int value2= calcScore(img, Point(prev.x + 5, prev.y), Point(x + 5, prev.y - STEP), 1, trash);

			int sum = value * 4;

			if (sum > maxSum)
			{
				maxSum = sum;
				maxPoint = maxPointT;
				lastAngle = atan((x - prev.x)/(STEP - prev.y));
			}
		}

		//std::cout << maxSum << maxPoint << std::endl;
		if (maxSum == 0) break;
		//cout << lastAngle * 180.0 / 3.14 << " " << maxPoint << prev << endl;
		prev = maxPoint;
		curve.push_back(maxPoint);

		fromX = tan(lastAngle - 3.0 * 3.14 / 180.0) * STEP;
		toX = tan(lastAngle + 3.0 * 3.14 / 180.0) * STEP;
	}

	if (curve.size() > 3)
	{
		if (curve[3].y < 350) return 0;
		std::vector<Point2f> newCurve;
		fitPointsYX(curve, newCurve, p1);
		curve = newCurve;
	}

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

int CurveDetector3::calcScore(const cv::UMat& img, Point a, Point b, float d, Point &maxPoint)
{
	int maxVal = 0;
	int res = 0;
	if (a == b) return 0;
	float slope = (float)(b.y - a.y) / (b.x - a.x);
	int dirx = 0;
	int diry = 0;
	int swapped = 0;
	Point startPoint = a;
	if (abs(slope) < 1)
	{
		dirx = 1;
		if (a.x > b.x)
		{
			swap(a, b);
			swapped = 1;
		}
	}
	else
	{
		diry = 1;
		if (a.y > b.y)
		{
			swap(a, b);
			swapped = 1;
		}
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

	int sum = 0;
	const int WIDTH = 17;
	Point arr[WIDTH];
	int values[WIDTH];

	for (int i = 0; i < WIDTH; i++) values[i] = 0;
	int it= 0;


	for (int i = from; i <= to; i++, p+=slope)
	{
		for (int j = p - range; j <= p + range; j++, counter++)
		{
			// TODO: optimize - make two loops (dirx=0 and dirx=1)
			Point pos(dirx * i + diry * j, dirx * j + diry * i);
			/*if (isPointOutOfRange(pos, img.cols, img.rows)){
				counter--;
				continue;
			}*/

			int value = img.getMat(ACCESS_READ).at<uchar>(pos);
			//if ((     (((i - from) > 30) && swapped) || (((to - i) > 30) && !swapped) ) && (value > maxVal))

			Point pt(dirx * i + diry * j, dirx * j + diry * i);

			Point lenVec = pt - startPoint;
			float len = lenVec.x * lenVec.x + lenVec.y * lenVec.y;

				sum = sum + value - values[it];

				values[it] = value;
				arr[it] = pt;
				it = (it + 1) % WIDTH;
				if (len > 100*100)
				{
					if (sum > maxVal)
					{
						maxPoint = arr[(it + WIDTH/2)%WIDTH];
						maxVal = sum ;
					}
				}
				res += value;
		}
	}

	return res / counter;
}

