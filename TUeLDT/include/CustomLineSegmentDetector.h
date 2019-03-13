/*
 * CustomLineSegmentDetector.h
 *
 *  Created on: Mar 13, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#ifndef TUELDT_CustomLINESEGMENTDETECTOR_H_
#define TUELDT_CustomLINESEGMENTDETECTOR_H_

#include "opencv2/opencv.hpp"
#include <vector>
#include "LineSegment.h"

// TODO Change name of the class - LineSegmentDetector is taken by openCV

class CustomLineSegmentDetector {
public:
	CustomLineSegmentDetector(int width, int height);
	bool run(cv::Mat img);
	virtual ~CustomLineSegmentDetector();
	int calcScore(const cv::Mat& img, cv::Point a, cv::Point b);

	double* mImgD;
	int* mImgI;
	size_t mSize;
	size_t mWidth;
	size_t mHeight;

	double quant; /*<! Bound to the quantization error on the gradient norm.
                       Example: if gray levels are quantized to integer steps,
                       the gradient (computed by finite differences) error
                       due to quantization will be bounded by 2.0, as the
                       worst case is when the error are 1 and -1, that
                       gives an error of 2.0.
                       Suggested value: 2.0         					*/
	double angTh; /*<! Gradient angle tolerance in the region growing
                       algorithm, in degrees.
                       Suggested value: 22.5 							*/
	double logEps; /*<!  Detection threshold, accept if -log10(NFA) > log_eps.
                       The larger the value, the more strict the detector is,
                       and will result in less detections.
                       (Note that the 'minus sign' makes that this
                       behavior is opposite to the one of NFA.)
                       The value -log10(NFA) is equivalent but more
                       intuitive than NFA:
                       - -1.0 gives an average of 10 false detections on noise
                       -  0.0 gives an average of 1 false detections on noise
                       -  1.0 gives an average of 0.1 false detections on nose
                       -  2.0 gives an average of 0.01 false detections on noise
                       .
                       Suggested value: 0.0 							*/
	double densityTh; /*<! Minimal proportion of 'supporting' points in a rectangle.
                       Suggested value: 0.7 							*/
	int nBins; 		/*<! Number of bins used in the pseudo-ordering of gradient
                       modulus.
                       Suggested value: 1024             				*/

	std::vector<LineSegment> seg;

};

#endif /* TUELDT_CustomLINESEGMENTDETECTOR_H_ */
