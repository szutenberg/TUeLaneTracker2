/*
 * LineSegment.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: msz
 */

#include "LineSegment.h"



bool operator<(const LineSegment& l1, const LineSegment& l2)
{
    return l1.a.y < l2.a.y;
}

bool operator==(const LineSegment& l1, const LineSegment& l2)
{
    return ((l1.a == l2.a) && (l1.b == l2.b) && (l1.score == l2.score) && (l1.angle == l2.angle));
}

bool operator!=(const LineSegment& l1, const LineSegment& l2)
{
    return !(l1 == l2);
}

std::ostream& operator<<(std::ostream& os, const LineSegment& d)
{
	cv::Point2f vec = d.a - d.b;
	int len = sqrt(vec.x * vec.x + vec.y * vec.y);
	cv::Point a = d.a;
	cv::Point b = d.b;
    os << a << "\t" << b << "\t" << (int)d.score << "\t" << d.NFA << "\t" << d.p << "\t" << d.width << "\t" << len << "\n";
    return os;
}


bool compareScores(const LineSegment& l1, const LineSegment& l2)
{
    return (l1.score > l2.score);
}
