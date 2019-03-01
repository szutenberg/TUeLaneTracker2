/*
 * Benchmark.cpp
 *
 *  Created on: Mar 1, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */



#include "Benchmark.h"
#include "Helpers.h"

Benchmark::Benchmark(cv::String path, const LaneTracker::Config& Config):
mConfig(Config),
mPtrFrameFeeder(nullptr),
mPtrFrameRenderer(nullptr),
mPtrLaneFilter(nullptr),
mPtrVanishingPtFilter(nullptr),
mPtrTemplates(nullptr),
mPtrLaneModel(nullptr)
{
	for (int val = 240; val <= 710; val+= 10) h_samples.push_back(val);

	mPtrBootingState   = unique_ptr<InitState>(new InitState(mConfig));
	mPtrLaneFilter.reset(nullptr);
	mPtrVanishingPtFilter.reset(nullptr);
	mPtrTemplates.reset(nullptr);

	mPtrLaneFilter 	        = mPtrBootingState->createLaneFilter();
	mPtrVanishingPtFilter   = mPtrBootingState->createVanishingPtFilter();
	mPtrTemplates 	        = mPtrBootingState->createTemplates();
	assert(mPtrBootingState->currentStatus == StateStatus::DONE);

	cv::String pathImg = path + "/20.jpg";

	glob(pathImg, mTestPaths, 1);
	for (auto& str : mTestPaths)
	{
		size_t pos = str.rfind('/');
		assert(pos != cv::String::npos);
		str = str.substr(0, pos);
	}
	Helpers::sortFileNames(mTestPaths);
	printf("Detected %lu sequences.\n", mTestPaths.size());


}

Benchmark::~Benchmark() {
	// TODO Auto-generated destructor stub
}


int Benchmark::run()
{
	for (cv::String testPath : mTestPaths)
	{
		printf("%s\n", testPath.c_str());

		mPtrFrameFeeder = unique_ptr<FrameFeeder>(new ImgStoreFeeder(testPath));
	    mPtrBufferingState.reset(new BufferingState<BufferingDAG_generic>(mConfig));
		mPtrBootingState = nullptr;
	    mPtrBufferingState->setupDAG(std::ref(*mPtrTemplates), mConfig.buffer_count);
	    mPtrFrameFeeder->Paused.store(false);

	    mPtrBufferingState->run(mPtrFrameFeeder->dequeue());
	    mPtrBufferingState->run(mPtrFrameFeeder->dequeue());
	    mPtrTrackingState.reset(new TrackingLaneState<TrackingLaneDAG_generic>( move(mPtrBufferingState->mGraph) ));
	    mPtrBufferingState 	= nullptr; //BufferingState does not contain graph anymore, so make it unusable.

	    mPtrTrackingState->setupDAG(mPtrLaneFilter.get(), mPtrVanishingPtFilter.get());
	    mPtrFrameRenderer.reset(new FrameRenderer(*mPtrLaneFilter, *mPtrFrameFeeder.get() ));

		for (int i = 1; i < 18; i++)
		{
			   mPtrLaneModel = mPtrTrackingState->run(mPtrFrameFeeder->dequeue());
			   mPtrFrameFeeder->dequeueDisplay();
		}

		mPtrLaneModel = mPtrTrackingState->run(mPtrFrameFeeder->dequeue());
		mPtrFrameRenderer->drawLane(mPtrFrameFeeder->dequeueDisplay(), *mPtrLaneModel);
	}

	return 0;
}
