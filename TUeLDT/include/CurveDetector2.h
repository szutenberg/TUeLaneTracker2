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
class CurveDetector2 {


public:
	CurveDetector2(const LaneTracker::Config* cfg, const LaneFilter* laneFilter, const VanishingPtFilter* vpFilter, const Templates* templates);
	virtual ~CurveDetector2();
	int run(TrackingLaneDAG_generic& tr, LaneModel* Lane);
	const LaneTracker::Config* mCfg;
	const LaneFilter* mLaneFilter;
	const VanishingPtFilter* mVpFilter;
	const Templates* mTemplates;
};

#endif /* TUELDT_CURVEDETECTOR2_H_ */
