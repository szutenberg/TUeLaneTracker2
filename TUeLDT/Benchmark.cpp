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

vector<cv::Point> Benchmark::generateHSamplesPoints(vector<cv::Point>& in)
{


	int i = in.size() - 1;
	vector<cv::Point> out;

	if (i <= 1) return out;

	for (int ypos : h_samples)
	{
		while ((in[i-1].y < ypos) && (i > 1))
		{
			i--;
		}

		cv::Point2f vec(in[i-1] - in[i]);
		cv::Point pos(in[i]);
		pos.x += ((float)(in[i-1].x - in[i].x)) * (ypos - in[i].y) / (in[i-1].y - in[i].y) + 0.5;
		if (pos.x < 0) pos.x = -2;
		if (pos.x > (mConfig.cam_res_h - 1)) pos.x = -2;
		pos.y = ypos;

		out.push_back(pos);
	}

	return out;
}


int Benchmark::run()
{
	for (cv::String testPath : mTestPaths)
	{
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


		printf("{\"lanes\": [");

		vector<cv::Point> dl, dr;
		dl = generateHSamplesPoints(mPtrLaneModel->curveLeft);
		dr = generateHSamplesPoints(mPtrLaneModel->curveRight);

		if (dl.size() == h_samples.size())
		{
			printf("[");
			for (size_t i = 0; i < dl.size(); i++)
			{
				if (i) printf(", ");
				printf("%d", dl[i].x);
			}
			printf("]");
			if (dr.size() == h_samples.size()) printf(",");
		}

		if (dr.size() == h_samples.size())
		{
			printf("[");
			for (size_t i = 0; i < dr.size(); i++)
			{
				if (i) printf(", ");
				printf("%d", dr[i].x);
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
