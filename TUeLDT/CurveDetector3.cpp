/******************************************************************************
* Copyright (c) 2019 Michal Szutenberg
* ****************************************************************************/

#include "CurveDetector3.h"
#include <Eigen/QR>
#include <lsd_1.6/lsd.h>
#include <lsd_1.6/lsd.c> // fix me

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
		if (p.y < 100) continue;
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
	Mat debugBird;
    img.copyTo(debugBird);
    cvtColor(debugBird, debugBird, COLOR_GRAY2BGR);

	int * n_out = new int;
	double * imgD = new double[img.cols * img.rows];

	if (img.isContinuous())
	{
		uchar * ptr = img.data;
		for (int i = 0; i < img.cols * img.rows; i++)
			imgD[i] = ptr[i];
	}
	else printf("Not continous!\n");

	double * ptr = LineSegmentDetection(n_out, imgD, img.cols, img.rows, 1, 1, 5, 22.5, 0, 0.5, 1024, NULL, NULL, NULL);
	//printf("%d\n", *n_out);

	vector<Point2f> l1, l2;
	vector<double> scores;
	vector<float> angle;
	double maxScore = 0;
	/* Get params - TODO - move to another class */
	for (int i = 0; i < *n_out; i++)
	{
		Point2f p1(ptr[i*7+0], ptr[i*7+1]);
		Point2f p2(ptr[i*7+2], ptr[i*7+3]);
		if (p1.y < p2.y) swap(p1, p2);
		Point trash;
		int score = calcScore(img, Point(p1+Point2f(0.5, 0.5)), Point(p2+Point2f(0.5, 0.5)), 1, trash);
		Point2f vec = p2 - p1;
		float angleVal = atan(vec.x / vec.y) * 180.0 / 3.14;
		scores.push_back(score);
		l1.push_back(p1);
		l2.push_back(p2);
		angle.push_back(angleVal);
		if (score > maxScore) maxScore = score;
	}

	maxScore /= 100.0;
	for (size_t i = 0; i < scores.size(); i++)
	{
		scores[i] /= maxScore; // now range from 0 to 100
		line(debugBird, l1[i], l2[i], CvScalar(0, scores[i] * 2, 0), 2);
	}

	int iSt = 0;
	float maxD = 0.1;
	for (size_t i = 0; i < l1.size(); i++)
	{
		if (abs(l1[i].x - start.x) > 25) continue;
		if (abs(angle[i]) > 10) continue;

		Point2f dst = l1[i] - Point2f(start);
		float dif = (dst.x * dst.x + dst.y * dst.y);
		if (dif < 10) dif = 10;
		float d = scores[i] * scores[i] / dif;

		if (maxD < d)
		{
			maxD = d;
			iSt = i;
		}
	}
	//cout << start << l2[iSt] << endl;

	curve.push_back(l1[iSt]);
	curve.push_back(l2[iSt]);

	//line(debugBird, l1[iSt], l2[iSt], CvScalar(200, 200, 0), 2);

	int color = 255;
	int best;

	do
	{
		Point2f vecSt = l2[iSt] - l1[iSt];

		float maxD = 0;
		best = -1;

		for (size_t i = 0; i < l1.size(); i++)
		{
			if (l1[i].y >= l2[iSt].y) continue;
			if (abs(angle[i] - angle[iSt]) > 12) continue;

			Point2f vecD = l1[i] - l2[iSt];
			if (abs(atan(vecD.x / vecD.y) - atan(vecSt.x / vecSt.y))  > (5.0 * 3.14 / 180.0)) continue;

			Point2f dst = l1[i] - l2[iSt];
			float dif = (dst.x * dst.x + dst.y * dst.y);
			if (dif < 10) dif = 10;
			float d = scores[i] * scores[i] / dif;

			if (maxD < d)
			{
				maxD = d;
				best = i;
			}
		}

		//cout << maxD << "\t" << best << endl;
		if (maxD < 0.1) best = -1;
		if (best != -1)
		{
			color -= 30;
			//line(debugBird, l1[best], l2[best], CvScalar(color, color, color), 2);
			iSt = best;


			Point2f vec = l2[best] - l1[best];

			float minLen = 5 * 5;
			int found = -1;
			for (int i = 0; i < (int)l1.size(); i++)
			{
				if (i == best) continue;
				if (abs(angle[i] - angle[best]) > 2) continue;
				Point2f dst = l1[i] - l1[best];
				float len = dst.x * dst.x + dst.y * dst.y;

				if (minLen > len)
				{
					minLen = len;
					found = i;
				}
			}

			vector<int> linesToAdd;
			linesToAdd.push_back(best);
			if (found != -1) linesToAdd.push_back(found);


			//cout << "For " << l1[best] << l2[best] << angle[best] << endl;
			//if (found != -1) cout << "found " << l1[found] << l2[found] << angle[found] << "    " << minLen << endl;


			for (int ind : linesToAdd)
			{
				curve.push_back(l1[ind]);
				vec = l2[ind] - l1[ind];

				if (l2[ind].y > 0)
				{
					int segments = sqrt(vec.x * vec.x + vec.y * vec.y) * scores[ind] / (700-l1[ind].y) / 4.0;
					if (segments > 1)
					{
						segments -=1;
						vec /= (float)segments;

						for (int i = 1; i < segments; i++) curve.push_back(l1[ind] + (float)i * vec);
					}
				}
				curve.push_back(l2[ind]);
			}
		}
	}while(best != -1);

	cv::Point c1(-3, 0);
	cv::Point c2(3, 0);
	cv::Point c3(0, 3);
	cv::Point c4(0, -3);
	for (Point p : curve)
	{
		line(debugBird, p + c1, p + c2, CvScalar(0, 0, 200), 2);
		line(debugBird, p + c3, p + c4, CvScalar(0, 0, 200), 2);
	}
	   imshow(name.c_str(), debugBird);


	if (curve.size() > 3)
	{
		if (curve[3].y < 350) return 0;
		std::vector<Point2f> newCurve;
		fitPointsYX(curve, newCurve, start);
		curve = newCurve;
	}




	return 0;
}



inline int CurveDetector3::isPointOutOfRange(Point a, int width, int height)
{
	return ((a.x < 1) || (a.y < 1) || (a.x > (width - 1)) || (a.y > (height - 1)));
}


int CurveDetector3::calcScore(const cv::Mat& img, Point a, Point b, float d, Point &maxPoint)
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

			int value = img.at<uchar>(pos);
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

