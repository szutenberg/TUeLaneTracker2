/*
 * LineSegment.h
 *
 *  Created on: Mar 13, 2019
 *      Author: msz
 */

#ifndef TUELDT_LINESEGMENT_H_
#define TUELDT_LINESEGMENT_H_

#include "opencv2/opencv.hpp"
#include <ostream>


typedef struct {
   cv::Point2f a;
   cv::Point2f b;
   float score;
   float angle;
   float NFA; // just for debug purposes
   float p;
   float width;
} LineSegment;


bool operator<(const LineSegment& l1, const LineSegment& l2);
bool operator==(const LineSegment& l1, const LineSegment& l2);

bool operator!=(const LineSegment& l1, const LineSegment& l2);
std::ostream& operator<<(std::ostream& os, const LineSegment& d);
bool compareScores(const LineSegment& l1, const LineSegment& l2);

#endif /* TUELDT_LINESEGMENT_H_ */
