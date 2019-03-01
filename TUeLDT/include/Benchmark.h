/*
 * Benchmark.h
 *
 *  Created on: Mar 1, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#ifndef TUELDT_BENCHMARK_H_
#define TUELDT_BENCHMARK_H_


#include<string>
#include<vector>
#include "opencv2/opencv.hpp"

#include "State.h"
#include "InitState.h"
#include "BufferingState.h"
#include "TrackingLaneState.h"
#include "FrameFeeder.h"
#include "FrameRenderer.h"

using namespace std;

class Benchmark {

public:
	Benchmark(cv::String path, const LaneTracker::Config& Config);
	virtual ~Benchmark();
	const LaneTracker::Config& 		mConfig;
	int run();


//private:
	 unique_ptr<InitState>						mPtrBootingState;

	unique_ptr<FrameFeeder>			mPtrFrameFeeder;
	unique_ptr<FrameRenderer>		mPtrFrameRenderer;

	unique_ptr<LaneFilter>  		mPtrLaneFilter;
	unique_ptr<VanishingPtFilter>  	mPtrVanishingPtFilter;
	unique_ptr<Templates> 			mPtrTemplates;
	LaneModel*						mPtrLaneModel;
    unique_ptr<BufferingState<BufferingDAG_generic>>  	 	mPtrBufferingState;
	 unique_ptr<TrackingLaneState<TrackingLaneDAG_generic>>   	mPtrTrackingState;


	vector< vector<  vector<cv::Point> > > mLanes; // lanes[Test][lane_id][point]
	vector< cv::String > mTestPaths;
	vector< int > h_samples;
};

#endif /* TUELDT_BENCHMARK_H_ */
