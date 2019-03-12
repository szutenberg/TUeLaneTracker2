/*
 * BirdView.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#include "BirdView.h"

BirdView::BirdView(){

}


inline double BirdView::det(double a, double b, double c, double d)
{
	return a*d - b*c;
}

Point2f BirdView::findCrossPoint(Point a1, Point a2, Point b1, Point b2)
{
	Point2f ret(0,0);

	double detL1 = det(a1.x, a1.y, a2.x, a2.y);
	double detL2 = det(b1.x, b1.y, b2.x, b2.y);

	double xnom = det(detL1, a1.x - a2.x, detL2, b1.x - b2.x);
	double ynom = det(detL1, a1.y - a2.y, detL2, b1.y - b2.y);
	double denom = det( a1.x - a2.x, a1.y - a2.y, b1.x - b2.x, b1.y - b2.y);

	if(abs(denom) > 0.0001)
	{
		ret.x = xnom / denom;
		ret.y = ynom / denom;
	}

	return ret;
}


bool BirdView::configureTransform(Point l1, Point l2, Point r1, Point r2, int maxH, int width, int height)
{
	Point2f crossPoint = findCrossPoint(l1, l2, r1, r2);

	if (crossPoint.x == 0) return false;

	Point2f vl = crossPoint - Point2f(l1);
	Point2f vr = crossPoint - Point2f(r1);

	float dl_len = sqrt(vl.x*vl.x + vl.y*vl.y);
	float dr_len = sqrt(vr.x*vr.x + vr.y*vr.y);

	float len = dl_len;
	if (dr_len < len) len = dr_len;

	vl /= dl_len;
	vr /= dr_len;

	vl *= -len;
	vr *= -len;

	vl += crossPoint;
	vr += crossPoint;

	Point2f cc = (vr + vl) / 2.0;

	Point2f tmpV = crossPoint - cc;
	tmpV *= 0.95;

	float tmpH = sqrt(tmpV.x*tmpV.x + tmpV.y*tmpV.y);
	//std::cout << "tmpH = " << tmpH << std::endl;
	if (tmpH > maxH) tmpV /= tmpH / maxH;

	Point2f dd = cc + tmpV;

	Point2f kl = findCrossPoint(l1, l2, dd, dd + (vr - vl));
	Point2f kr = findCrossPoint(r1, r2, dd, dd + (vr - vl));

	Point2f el = kl - (kr - kl) * 3.0;
	Point2f er = kr + (kr - kl) * 3.0;

	Point2f bl = vl - (vr - vl) * 3.0;
	Point2f br = vr + (vr - vl) * 3.0;

    // Note that points in inputQuad and outputQuad have to be from top-left in clockwise order
    mInputQuad[0] = el;
    mInputQuad[1] = er;
    mInputQuad[2] = br;
    mInputQuad[3] = bl;

    mWidth = width;
    mHeight = height;

    mOutputQuad[0] = Point2f(0, 0);
    mOutputQuad[1] = Point2f(width - 1, 0);
    mOutputQuad[2] = Point2f(width - 1, height - 1);
    mOutputQuad[3] = Point2f(0, height - 1);

    mLambda = getPerspectiveTransform( mInputQuad, mOutputQuad );

	invert(mLambda, mLambdaInv);

	return true;
}


Mat BirdView::applyTransformation(Mat img)
{
	Mat output;
    warpPerspective(img, output, mLambda, Size(mWidth, mHeight) );
    return output;
}


bool BirdView::invertPoints(std::vector<Point2f>& in, std::vector<Point2f> &out)
{
	perspectiveTransform(in, out, mLambdaInv);

	return true;
}

BirdView::~BirdView() {
	// TODO Auto-generated destructor stub
}

