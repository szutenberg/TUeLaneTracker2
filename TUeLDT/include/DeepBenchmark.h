/*
 * Benchmark.h
 *
 *  Created on: Mar 1, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */

#ifndef TUELDT_DEEPBENCHMARK_H_
#define TUELDT_DEEPBENCHMARK_H_


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

class DeepBenchmark {

public:
	DeepBenchmark(cv::String path, const LaneTracker::Config& Config);
	virtual ~DeepBenchmark();
	const LaneTracker::Config& 		mConfig;
	int run();
	vector<cv::Point2f> generateHSamplesPointsFloat(vector<cv::Point2f>& in);
	vector<cv::Point> generateHSamplesPoints(vector<cv::Point2f>& in);

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
	vector< cv::String > mImgPaths;
	vector< int > h_samples;
};

#endif /* TUELDT_DEEPBENCHMARK_H_ */
