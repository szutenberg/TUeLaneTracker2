/*
 * CurveDetector2.h
 *
 *  Created on: Jun 11, 2019
 *      Author: Michal Szutenberg
 */

#ifndef TUELDT_CURVEDETECTOR2_H_
#define TUELDT_CURVEDETECTOR2_H_

#include "Config.h"
#include "LaneModel.h"
#include "Templates.h"
#include "BirdView.h"
#include "TrackingLaneDAG_generic.h"
#include "MapGenerator.h"

class CurveDetector2 {


public:
	CurveDetector2(const LaneTracker::Config* cfg, const LaneFilter* laneFilter, const VanishingPtFilter* vpFilter, const Templates* templates);
	virtual ~CurveDetector2();
	int run(TrackingLaneDAG_generic& tr, LaneModel* Lane, Mat input);
	const LaneTracker::Config* mCfg;
	const LaneFilter* mLaneFilter;
	const VanishingPtFilter* mVpFilter;
	const Templates* mTemplates;


	struct ProbMap buf[5];
	int bufIt;
	int bufSize;
	struct ProbMap finalMap;
	MapGenerator mapGen;
	cv::Rect ROI;
	cv::Mat mMask;
	cv::Mat ransac(Point2f base, Point2f pur, int rightSide, int N, Point2f pts[][12], Point2f crPt[][7], float result[]);
	cv::Mat preparePotentialBoundaries(Point2f base, Point2f pur);

	Mat timeline;
	Mat timeline2;

	int timelineIt;

};

#endif /* TUELDT_CURVEDETECTOR2_H_ */
