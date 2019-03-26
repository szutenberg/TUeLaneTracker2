/*
 * BirdView.h
 *
 *  Created on: Mar 7, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#ifndef TUELDT_BIRDVIEW_H_
#define TUELDT_BIRDVIEW_H_

#include<vector>
#include "opencv2/opencv.hpp"
using namespace cv;

class BirdView {

private:
	Mat mLambda;
	Mat mLambdaInv;

    // Input Quadilateral or Image plane coordinates
    Point2f mInputQuad[4];
    // Output Quadilateral or World plane coordinates
    Point2f mOutputQuad[4];

    int mWidth;
    int mHeight;

public:
	BirdView();
	virtual ~BirdView();

	bool configureTransform(Point l1, Point l2, Point r1, Point r2, int maxH, int width, int height);
	bool configureTransform2(Point l1, Point r1, Point vp, int width, int height);

	Mat applyTransformation(Mat img);
	bool invertPoints(std::vector<Point2f>& in, std::vector<Point2f> &out);
	bool convertPointsToBird(std::vector<Point2f>& in, std::vector<Point2f> &out);
	static double det(double a, double b, double c, double d);

	static Point2f findCrossPoint(Point a1, Point a2, Point b1, Point b2);

};

#endif /* TUELDT_BIRDVIEW_H_ */
