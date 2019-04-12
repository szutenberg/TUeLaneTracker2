/*
 * DeepBenchmark.cpp
 *
 *  Created on: Apr 2, 2019
 *      Author: Michal Szutenberg <michal@szutenberg.pl>
 */



#include "DeepBenchmark.h"
#include "Helpers.h"

extern int debugX;

DeepBenchmark::DeepBenchmark(cv::String path, const LaneTracker::Config& Config):
mConfig(Config),
mPtrFrameFeeder(nullptr),
mPtrFrameRenderer(nullptr),
mPtrLaneFilter(nullptr),
mPtrVanishingPtFilter(nullptr),
mPtrTemplates(nullptr),
mPtrLaneModel(nullptr)
{
	for (int val = 330; val <= 480; val+= 50) h_samples.push_back(val);

	mPtrBootingState   = unique_ptr<InitState>(new InitState(mConfig));
	mPtrLaneFilter.reset(nullptr);
	mPtrVanishingPtFilter.reset(nullptr);
	mPtrTemplates.reset(nullptr);

	mPtrLaneFilter 	        = mPtrBootingState->createLaneFilter();
	mPtrVanishingPtFilter   = mPtrBootingState->createVanishingPtFilter();
	mPtrTemplates 	        = mPtrBootingState->createTemplates();
	assert(mPtrBootingState->currentStatus == StateStatus::DONE);

	cv::String pathImg = path;
	glob(pathImg, mImgPaths, 1);
	sort(mImgPaths.begin(), mImgPaths.end());

	size_t pos = mImgPaths[0].rfind('/');
	assert(pos != cv::String::npos);
	cv::String lastPath = mImgPaths[0].substr(0, pos);

	mTestPaths.push_back(lastPath);
}

DeepBenchmark::~DeepBenchmark() {
	// TODO Auto-generated destructor stub
}

vector<cv::Point> DeepBenchmark::generateHSamplesPoints(vector<cv::Point2f>& in)
{
	vector<cv::Point2f> tmp = generateHSamplesPointsFloat(in);
	vector<cv::Point> out;

	for (int i = 0; i < tmp.size(); i++) out.push_back(cv::Point(tmp[i]));

	return out;
}


vector<cv::Point2f> DeepBenchmark::generateHSamplesPointsFloat(vector<cv::Point2f>& in)
{
	int i = in.size() - 1;
	vector<cv::Point2f> out;

	if (i < 1) return out;

	for (int ypos : h_samples)
	{
		while ((in[i-1].y < ypos) && (i >= 1))
		{
			i--;
		}
		//cout << ypos << "\t" << in[i-1] << "\t" << in[i] << endl;
		cv::Point2f vec(in[i-1] - in[i]);
		cv::Point2f pos(in[i]);
		pos.x += ((float)(in[i-1].x - in[i].x)) * ((float)ypos - in[i].y) / (in[i-1].y - in[i].y);
		//cout << pos.x << endl;
		pos.y = ypos;


		out.push_back(pos);
	}

	return out;
}



int DeepBenchmark::run()
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

	    // i have read already two frames

	    for (int i = 2; i <= 4; i++)
	    {
	    	frame = mPtrFrameFeeder->dequeue();
	    	mPtrLaneModel = mPtrTrackingState->run(frame);

	    	display = mPtrFrameFeeder->dequeueDisplay();

	    	mPtrLaneModel->benchL = generateHSamplesPoints(mPtrLaneModel->curveL);
	    	mPtrLaneModel->benchR = generateHSamplesPoints(mPtrLaneModel->curveR);
	    	if (debugX == 0) mPtrFrameRenderer->drawLane(display, *mPtrLaneModel);
	    }

	    for (int i = 5; i < mImgPaths.size(); i++)
	    {
			frame = mPtrFrameFeeder->dequeue();
			mPtrLaneModel = mPtrTrackingState->run(frame);

			display = mPtrFrameFeeder->dequeueDisplay();

			mPtrLaneModel->benchL = generateHSamplesPoints(mPtrLaneModel->curveL);
			mPtrLaneModel->benchR = generateHSamplesPoints(mPtrLaneModel->curveR);

			vector<cv::Point2f> benchLfloat = generateHSamplesPointsFloat(mPtrLaneModel->curveL);
			vector<cv::Point2f> benchRfloat = generateHSamplesPointsFloat(mPtrLaneModel->curveR);


			cv::Point2f r1, r2, l1, l2;
			r1.y = l1.y = 480;
			r2.y = l2.y = 360; ///// Watch out

			r1.x = mPtrLaneModel->boundaryRight[0] + 320;
			r2.x = mPtrLaneModel->boundaryRight[1] + 320;

			l1.x = mPtrLaneModel->boundaryLeft[0] + 320;
			l2.x = mPtrLaneModel->boundaryLeft[1] + 320;

			cv::Point2f vr = r2 - r1;
			cv::Point2f vl = l2 - l1;
			vr /= r1.y-r2.y;
			vl /= r1.y-r2.y;



			if (debugX == 0) mPtrFrameRenderer->drawLane(display, *mPtrLaneModel);

			printf("{\"lanes\": [");

			if (mPtrLaneModel->benchL.size() == h_samples.size())
			{
				printf("[");
				for (size_t i = 0; i < mPtrLaneModel->benchL.size(); i++)
				{
					if (i) printf(", ");
					printf("%4.1f", benchLfloat[i].x);
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
					printf("%4.1f", benchRfloat[i].x);
				}
				printf("]");
			}
			printf("]");

			printf(", \"lanesB\": [[");

			for (size_t i = 0; i < h_samples.size(); i++)
			{
				float x = l1.x + vl.x * (l1.y - h_samples[i]);
				if (i) printf(", ");
				printf("%4.1f", x);
			}
			printf("], [");
			for (size_t i = 0; i < h_samples.size(); i++)
			{
				float x = r1.x + vr.x * (r1.y - h_samples[i]);
				if (i) printf(", ");
				printf("%4.1f", x);
			}
			printf("]]");

			printf(", \"h_samples\": [");
			for (size_t i = 0; i < h_samples.size(); i++)
			{
				if (i) printf(", ");
				printf("%d", h_samples[i]);
			}
			cv::String name = mImgPaths[i].substr(mImgPaths[i].rfind('/')+1);
			printf("], \"raw_file\": \"%s\"}\n", name.c_str());

	    }
	    if (debugX == 0) cvWaitKey(100000);

	}

	return 0;
}
