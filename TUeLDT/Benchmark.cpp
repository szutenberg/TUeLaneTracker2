/*
 * Benchmark.cpp
 *
 *  Created on: Mar 1, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */



#include "Benchmark.h"
#include "Helpers.h"

extern int debugX;

Benchmark::Benchmark(cv::String path, const LaneTracker::Config& Config):
mConfig(Config),
mPtrFrameFeeder(nullptr),
mPtrFrameRenderer(nullptr),
mPtrLaneFilter(nullptr),
mPtrVanishingPtFilter(nullptr),
mPtrTemplates(nullptr),
mPtrLaneModel(nullptr)
{
	for (int val = 160; val <= 710; val+= 10) h_samples.push_back(val);

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
	// printf("Detected %lu sequences.\n", mTestPaths.size());


}

Benchmark::~Benchmark() {
	// TODO Auto-generated destructor stub
}

vector<cv::Point> Benchmark::generateHSamplesPoints(vector<cv::Point2f>& in, int ymin)
{
	//ymin = 470; // we focus only on points until purview line
	int i = in.size() - 1;
	vector<cv::Point> out;

	if (i < 1) return out;

	for (int ypos : h_samples)
	{
		while ((in[i-1].y < ypos) && (i >= 1))
		{
			i--;
		}

		cv::Point2f vec(in[i-1] - in[i]);
		cv::Point pos(in[i]);
		pos.x += ((float)(in[i-1].x - in[i].x)) * (ypos - in[i].y) / (in[i-1].y - in[i].y) + 0.5;
		if (pos.x < 0) pos.x = -2;
		if (pos.x > (mConfig.cam_res_h - 1)) pos.x = -2;
		pos.y = ypos;

		if (pos.y < ymin) pos.x = 10000;
		if (pos.y < ymin) pos.x = -2;

		out.push_back(pos);
	}

	return out;
}


int Benchmark::run()
{
	for (cv::String testPath : mTestPaths)
	{
		std::cerr << testPath << endl;
		mPtrFrameFeeder = unique_ptr<FrameFeeder>(new ImgStoreFeeder(testPath));
	    mPtrBufferingState.reset(new BufferingState<BufferingDAG_generic>(mConfig));
		mPtrBootingState = nullptr;
	    mPtrBufferingState->setupDAG(std::ref(*mPtrTemplates), mConfig.buffer_count);
	    mPtrFrameFeeder->Paused.store(false);

	    cv::UMat frame, display;

	    frame = mPtrFrameFeeder->dequeue();
	    mPtrBufferingState->run(frame);

	    frame = mPtrFrameFeeder->dequeue();
	    mPtrBufferingState->run(frame);

	    display = mPtrFrameFeeder->dequeueDisplay();
	    display = mPtrFrameFeeder->dequeueDisplay();

	    mPtrTrackingState.reset(new TrackingLaneState<TrackingLaneDAG_generic>( move(mPtrBufferingState->mGraph) ));
	    mPtrBufferingState 	= nullptr; //BufferingState does not contain graph anymore, so make it unusable.

	    mPtrTrackingState->setupDAG(mPtrLaneFilter.get(), mPtrVanishingPtFilter.get());
	    mPtrFrameRenderer.reset(new FrameRenderer(*mPtrLaneFilter, *mPtrFrameFeeder.get() ));

	    for (int i = 1; i <= 17; i++)
	    {
	    	frame = mPtrFrameFeeder->dequeue();
	    	mPtrLaneModel = mPtrTrackingState->run(frame);

	    	display = mPtrFrameFeeder->dequeueDisplay();

	    	mPtrLaneModel->benchL = generateHSamplesPoints(mPtrLaneModel->curveL, mPtrLaneModel->vanishingPt.V + mConfig.cam_res_v/2);
	    	mPtrLaneModel->benchR = generateHSamplesPoints(mPtrLaneModel->curveR, mPtrLaneModel->vanishingPt.V + mConfig.cam_res_v/2);
	    	if (debugX == 0) mPtrFrameRenderer->drawLane(display, *mPtrLaneModel);
	    }

    	frame = mPtrFrameFeeder->dequeue();
    	mPtrLaneModel = mPtrTrackingState->run(frame);

    	display = mPtrFrameFeeder->dequeueDisplay();

	    mPtrLaneModel->benchL = generateHSamplesPoints(mPtrLaneModel->curveL, mPtrLaneModel->vanishingPt.V + mConfig.cam_res_v/2);
	    mPtrLaneModel->benchR = generateHSamplesPoints(mPtrLaneModel->curveR, mPtrLaneModel->vanishingPt.V + mConfig.cam_res_v/2);

	    if (debugX == 0) mPtrFrameRenderer->drawLane(display, *mPtrLaneModel);
	    if (debugX == 0) cvWaitKey(100000);

		printf("{\"lanes\": [");



		if (mPtrLaneModel->benchL.size() == h_samples.size())
		{
			printf("[");
			for (size_t i = 0; i < mPtrLaneModel->benchL.size(); i++)
			{
				if (i) printf(", ");
				printf("%d", mPtrLaneModel->benchL[i].x);
			}
			printf("]");
			if (mPtrLaneModel->benchR.size() == h_samples.size()) printf(",");
		}

		if (mPtrLaneModel->benchR.size() == h_samples.size())
		{
			printf("[");
			for (size_t i = 0; i < mPtrLaneModel->benchR.size(); i++)
			{
				if (i) printf(", ");
				printf("%d", mPtrLaneModel->benchR[i].x);
			}
			printf("]");
		}
		printf("]");
		printf(", \"h_samples\": [");
		for (size_t i = 0; i < h_samples.size(); i++)
		{
			if (i) printf(", ");
			printf("%d", h_samples[i]);
		}
		printf("], \"raw_file\": \"%s%s\"}\n", testPath.c_str(), "/20.jpg");
	}

	return 0;
}
