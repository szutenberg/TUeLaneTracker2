/*
 * CurveDetector.h
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#ifndef TUELDT_CURVEDETECTOR_H_
#define TUELDT_CURVEDETECTOR_H_

#include "Config.h"
#include "LaneModel.h"


class CurveDetector {

private:
	const LaneTracker::Config* mCfg;

public:
	CurveDetector(const LaneTracker::Config* cfg);
	virtual ~CurveDetector();
	int run(cv::UMat& frame, LaneModel* Lane);
};

#endif /* TUELDT_CURVEDETECTOR_H_ */
