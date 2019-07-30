/*
* Copyright (c) 2017 NXP Semiconductor;
* All Rights Reserved
*
* AUTHOR : Rameez Ismail
*
* THIS SOFTWARE IS PROVIDED BY NXP "AS IS" AND ANY EXPRESSED OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL NXP OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
* ****************************************************************************/ 

#include "TrackingLaneDAG_generic.h"
#include "ScalingFactors.h"
using namespace cv;
TrackingLaneDAG_generic::TrackingLaneDAG_generic(BufferingDAG_generic&& bufferingGraph)
: 
  BufferingDAG_generic(std::move(bufferingGraph)),
  mMAX_PIXELS_ROI(mFrameGRAY_ROI.rows * mFrameGRAY_ROI.cols),
  mLikelihood_LB(0),
  mLikelihood_RB(0),
  mLikelihood_NB(0),
  mLikelihood_W(0),
  mConditionalProb(0),
  mPosterior(0),
  mMaxPosterior(0),
  mInvCorrelationNB(0),
  mLOWER_LIMIT_BASE(0),
  mLOWER_LIMIT_PURVIEW(0),
  mLOWER_LIMIT_FAR(0),
  mUPPER_LIMIT_BASE(0),
  mUPPER_LIMIT_PURVIEW(0),
  mUPPER_LIMIT_FAR(0),
  mSTEP_BASE_SCALED(0),
  mSTEP_PURVIEW_SCALED(0),
  mSTEP_FAR_SCALED(0),
  mIdxPurview_LB(0),
  mIdxPurview_RB(0)
{	

}

int previousModel = 0;
cv::Point lastPosL, lastPosR;
int trackingCnt = 0;

cv::Mat oldBaseHist;
cv::Mat mDepthTemplate2;

int TrackingLaneDAG_generic::init_DAG(LaneFilter* laneFilter, VanishingPtFilter* vpFilter)
{

	mLaneFilter		= laneFilter;
	mVpFilter		= vpFilter;

	mPtrLaneModel.reset(new LaneModel([=]()->vector<float>
					 {
						vector<float> lookAheadPts;
						lookAheadPts.push_back(laneFilter->BASE_LINE_cm);
						lookAheadPts.push_back(laneFilter->PURVIEW_LINE_cm);
						return lookAheadPts;
					 }()));


        mX_ICCS   		= mX_ICS + mCAMERA.O_ICS_ICCS.x;
        mY_ICCS   		= mY_ICS + mCAMERA.O_ICS_ICCS.y;

	const size_t& lCOUNT    = mLaneFilter->COUNT_BINS;

	mX_ICCS_SCALED	 	    =  SCALE_INTSEC*mX_ICCS;
	mBASE_BINS_SCALED  	    =  SCALE_INTSEC*mLaneFilter->BASE_BINS;
	mPURVIEW_BINS_SCALED	=  SCALE_INTSEC*mLaneFilter->PURVIEW_BINS;
	mFAR_BINS_SCALED	    =  SCALE_INTSEC*mLaneFilter->FAR_BINS;

	mSTEP_BASE_SCALED	    =  SCALE_INTSEC*mLaneFilter->BASE_STEP;
	mSTEP_PURVIEW_SCALED	=  SCALE_INTSEC*mLaneFilter->PURVIEW_STEP;
	mSTEP_FAR_SCALED	    =  SCALE_INTSEC*mLaneFilter->FAR_STEP;

	mLOWER_LIMIT_BASE	    =  mBASE_BINS_SCALED.at<int32_t>(0,0);
	mLOWER_LIMIT_PURVIEW  	=  mPURVIEW_BINS_SCALED.at<int32_t>(0,0);
	mLOWER_LIMIT_FAR  	    =  mFAR_BINS_SCALED.at<int32_t>(0,0);

	mUPPER_LIMIT_BASE	    =  mBASE_BINS_SCALED.at<int32_t>(lCOUNT-1,0);
	mUPPER_LIMIT_PURVIEW  	=  mPURVIEW_BINS_SCALED.at<int32_t>(lCOUNT-1,0);
	mUPPER_LIMIT_FAR  	    =  mFAR_BINS_SCALED.at<int32_t>(lCOUNT-1,0);


    mHistBase               =  cv::Mat::zeros(lCOUNT,  1 ,  CV_32S);
    mHistPurview            =  cv::Mat::zeros(lCOUNT,  1 ,  CV_32S);
    mHistFar                =  cv::Mat::zeros(lCOUNT,  1 ,  CV_32S);


  	return 0;
}

const int STAB_BUF_LEN = 5;
Point sBufR[STAB_BUF_LEN];
Point sBufL[STAB_BUF_LEN];
int sBufIt = 0;

int stabilityCounter = 0;
int tracking = 0;
const int DIR_MARGIN = 50;


void TrackingLaneDAG_generic::execute(cv::UMat& FrameGRAY)
{

	BufferingDAG_generic::execute(FrameGRAY);


#ifdef PROFILER_ENABLED
mProfiler.start("TEMPORAL_FILTERING");
#endif	

      #ifdef TEST_APEX_CODE
        mPROB_FRAME_0           = mBufferPool->Probability[0].clone();          // copy, otherwise overwritten by shift operation.
        mGRAD_FRAME_0           = mBufferPool->GradientTangent[0].clone();      // copy, otherwise overwritten by shift operation.
        mProbMapFocussed        = mBufferPool->Probability[0].clone();          // copy otherwise no test point for temporal pooling.
        mGradTanFocussed        = mBufferPool->GradientTangent[0].clone();      // copy otherwise no test point for temporal pooling.

      #else
        mProbMapFocussed        = mBufferPool->Probability[0];
        mGradTanFocussed        = mBufferPool->GradientTangent[0];
      #endif


      for ( std::size_t i=1; i< mBufferPool->Probability.size(); i++ )
      {
          mMask = mProbMapFocussed < mBufferPool->Probability[i];
          mBufferPool->Probability[i].copyTo(mProbMapFocussed, mMask );
          mBufferPool->GradientTangent[i].copyTo(mGradTanFocussed, mMask );
      }
	
      imshow("mProbMapFocussed", mProbMapFocussed);

#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Temporal Filtering [Max-Pooling]." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("TEMPORAL_FILTERING")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("TEMPORAL_FILTERING")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("TEMPORAL_FILTERING")<<endl
				<<"******************************"<<endl<<endl;	
				#endif





#ifdef PROFILER_ENABLED
mProfiler.start("COMPUTE_INTERSECTIONS");
#endif	
	//Note: Gradients are not computed in the Image-center-cs. slope is positive in upward direction.
	//Therefore invert the mY_ICCS by subtracting the BASE_LINE, histNorm[10000]
	//Weights of the intersections above the vanishingPt are already set to zero.

	//Base Intersections
	subtract(mY_ICCS, mLaneFilter->BASE_LINE_ICCS, mIntBase, cv::noArray(), CV_32S);
	divide(mIntBase, mGradTanFocussed, mIntBase, SCALE_INTSEC_TAN, CV_32S);
	add(mIntBase, mX_ICCS_SCALED, mIntBase);
	
	//Purview Intersections
	subtract(mY_ICCS, mLaneFilter->PURVIEW_LINE_ICCS, mIntPurview, cv::noArray(), CV_32S);
	divide(mIntPurview,mGradTanFocussed, mIntPurview, SCALE_INTSEC_TAN, CV_32S);
	add(mIntPurview, mX_ICCS_SCALED, mIntPurview);

	//Far Intersections
	subtract(mY_ICCS, mLaneFilter->FAR_LINE_ICCS, mIntFar, cv::noArray(), CV_32S);
	divide(mIntFar, mGradTanFocussed, mIntFar, SCALE_INTSEC_TAN, CV_32S);
	add(mIntFar, mX_ICCS_SCALED, mIntFar);

	bitwise_and(mBufferPool->Probability[mBufferPos], mFocusTemplate, mBufferPool->Probability[mBufferPos]);


#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Compute Intersections with Bottom and Horizon." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("COMPUTE_INTERSECTIONS")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("COMPUTE_INTERSECTIONS")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("COMPUTE_INTERSECTIONS")<<endl
				<<"******************************"<<endl<<endl;	
				#endif

		



#ifdef PROFILER_ENABLED
mProfiler.start("MASK_INVALID_BIN_IDS");
#endif
     {
	//Build Mask for Valid Intersections
        mMask =  mIntBase    >  mLOWER_LIMIT_BASE;
	bitwise_and(mMask, mIntPurview >  mLOWER_LIMIT_PURVIEW, mMask);
    	bitwise_and(mMask, mIntBase    <  mUPPER_LIMIT_BASE, 	mMask);
    	bitwise_and(mMask, mIntPurview <  mUPPER_LIMIT_PURVIEW, mMask);

	//^TODO: Put on the side thread
        mHistBase      = cv::Mat::zeros(mLaneFilter->COUNT_BINS,  1 ,  CV_32S);
        mHistPurview   = cv::Mat::zeros(mLaneFilter->COUNT_BINS,  1 ,  CV_32S);
     }		

#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Extract Valid Intersection Bin IDs." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("MASK_INVALID_BIN_IDS")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("MASK_INVALID_BIN_IDS")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("MASK_INVALID_BIN_IDS")<<endl
				<<"******************************"<<endl<<endl;	
				#endif 






#ifdef PROFILER_ENABLED
mProfiler.start("COMPUTE_HISTOGRAMS");
#endif
	//Weights of Intersections

	if (mDepthTemplate2.rows == 0)
	{
		mDepthTemplate2 = cv::Mat::zeros(mDepthTemplate.rows, mDepthTemplate.cols, CV_8U);

		for (int y = 0; y < mDepthTemplate.rows; y++)
		{
			double dy = mDepthTemplate.rows - y;
			double coeff = 50000;
			uint16_t val = exp(-dy*dy / coeff) * 255;

			for (int x = 0; x < mDepthTemplate.cols; x++)
			{
				mDepthTemplate2.at<uint8_t>(y, x) = val;
			}
		}
		imshow("mDepthTemplate2", mDepthTemplate2);

	}


	multiply(mDepthTemplate2, mProbMapFocussed, mIntWeights, 1/255.0, CV_32S);

	cv::Point min_loc, max_loc;
	double min, max;
	//cv::minMaxLoc(mIntWeights, &min, &max, &min_loc, &max_loc);
	//cout << "Max value: " << max << endl;
	cv::Mat hist2dr = cv::Mat::zeros(DIR_MARGIN*2 + 1, mHistBase.rows, CV_32S);
	cv::Mat hist2dl = cv::Mat::zeros(DIR_MARGIN*2 + 1, mHistBase.rows, CV_32S);

		int32_t* 	lPtrIntBase 	    = mIntBase.ptr<int32_t>(0);
		int32_t* 	lPtrIntPurview      = mIntPurview.ptr<int32_t>(0);
		int32_t* 	lPtrIntFar          = mIntFar.ptr<int32_t>(0);

		int32_t* 	lWeightBin   	    = mIntWeights.ptr<int32_t>(0);
		uint8_t* 	lPtrMask            = mMask.ptr<uint8_t>(0);

		int32_t* 	lPtrHistBase        =  mHistBase.ptr<int32_t>(0);
		int32_t* 	lPtrHistPurview     =  mHistPurview.ptr<int32_t>(0);
		int32_t* 	lPtrHistFar         =  mHistFar.ptr<int32_t>(0);

		int16_t   	lBaseBinIdx;
		int16_t   	lPurviewBinIdx;
		int16_t   	lFarBinIdx;

		mPtrLaneModel->candL.clear();
		mPtrLaneModel->candR.clear();
		mPtrLaneModel->valL.clear();
		mPtrLaneModel->valR.clear();

		for (int i = 0; i < mMAX_PIXELS_ROI; i++,lPtrIntBase++,lPtrIntPurview++, lPtrIntFar++, lWeightBin++ , lPtrMask++)
		{
			if(!(*lPtrMask ==0) )
			{
				lBaseBinIdx	= (*lPtrIntBase    - mLOWER_LIMIT_BASE    + (mSTEP_BASE_SCALED/2))/mSTEP_BASE_SCALED;
				lPurviewBinIdx	= (*lPtrIntPurview - mLOWER_LIMIT_PURVIEW + (mSTEP_PURVIEW_SCALED/2))/mSTEP_PURVIEW_SCALED;
				lFarBinIdx	= (*lPtrIntFar - mLOWER_LIMIT_FAR + (mSTEP_FAR_SCALED/2))/mSTEP_FAR_SCALED;

				assert( lBaseBinIdx < mHistBase.rows );

				*(lPtrHistBase       + lBaseBinIdx   )  	+= *lWeightBin;
				*(lPtrHistPurview    + lPurviewBinIdx)      += *lWeightBin;


				int16_t y = DIR_MARGIN + lPurviewBinIdx - lBaseBinIdx;
				int16_t x = lBaseBinIdx;

				if (y >= hist2dr.rows) continue;
				if (x >= hist2dr.cols) continue;
				if (x < 0) continue;
				if (y < 0) continue;

				if (lBaseBinIdx < mHistBase.rows/2) hist2dl.at<int32_t>(y, x) += *lWeightBin;
				else hist2dr.at<int32_t>(y, x) += *lWeightBin;

				if (lFarBinIdx < mHistBase.rows)
				{
					*(lPtrHistFar   + lFarBinIdx)      += *lWeightBin;
				}
			}
		}

        Mat histR, histL;
		double maxL, maxR;

		cv::minMaxLoc(hist2dl, &min, &maxL, &min_loc, &max_loc);
	    cv::minMaxLoc(hist2dr, &min, &maxR, &min_loc, &max_loc);

        hist2dr.convertTo(histR, CV_64F, 1.0/maxR);
        hist2dl.convertTo(histL, CV_64F, 1.0/maxL);


	    {
	        Mat kernel;

	        kernel = Mat::zeros( 3, 3, CV_64F );

			double sum = (1 + 2 + 4 + 2 + 1);
			kernel.at<double>(0,0) = 0;
			kernel.at<double>(0,1) = 1;
			kernel.at<double>(0,2) = 0;
			kernel.at<double>(1,0) = 2;
			kernel.at<double>(1,1) = 4;
			kernel.at<double>(1,2) = 2;
			kernel.at<double>(2,0) = 0;
			kernel.at<double>(2,1) = 1;
			kernel.at<double>(2,2) = 0;
			kernel /= sum;

	        filter2D(histR, histR, CV_64F , kernel);
	        filter2D(histL, histL, CV_64F , kernel);


	        kernel = Mat::zeros(1, 5, CV_64F);
	        kernel.at<double>(0, 0) = -1;
	        kernel.at<double>(0, 1) = -1;
	        kernel.at<double>(0, 2) = 2;
	        kernel.at<double>(0, 3) = 1;
	        kernel.at<double>(0, 4) = 1;

	        filter2D(histR, histR, CV_64F , kernel);
	        kernel.at<double>(0, 0) = 1;
	        kernel.at<double>(0, 1) = 1;
	        kernel.at<double>(0, 2) = 2;
	        kernel.at<double>(0, 3) = -1;
	        kernel.at<double>(0, 4) = -1;
	        filter2D(histL, histL, CV_64F , kernel);
	    }


        // Apply filter

    	cv::minMaxLoc(histL, &min, &maxL, &min_loc, &max_loc);
    	histL /= maxL;

    	cv::minMaxLoc(histR, &min, &maxR, &min_loc, &max_loc);
    	histR /= maxR;


    	Mat hist;
    	add(histL, histR, hist, noArray(), CV_32F);

    	imshow("hist Final", hist);

    	double totalMax = (maxL > maxR)?maxL:maxR;
		const int CAND_AMT = 20;

		cv::Point candR[CAND_AMT];
		double valR[CAND_AMT];
		cv::Point candL[CAND_AMT];
		double valL[CAND_AMT];

		for (int i = 0; i < CAND_AMT; i++)
		{
			cv::minMaxLoc(histR, &min, &max, &min_loc, &max_loc);

			//hist2dr.at<int32_t>(max_loc.y, max_loc.x) = 1;

			candR[i] = max_loc+ Point(0, max_loc.x - DIR_MARGIN);
			valR[i] = max/maxR;

			mPtrLaneModel->candR.push_back(max_loc + Point(0, max_loc.x - DIR_MARGIN));
			mPtrLaneModel->valR.push_back(valR[i]);
		}

		for (int i = 0; i < CAND_AMT; i++)
		{
			cv::minMaxLoc(histL, &min, &max, &min_loc, &max_loc);

			//hist2dl.at<int32_t>(max_loc.y, max_loc.x) = 1;

			candL[i] = max_loc+ Point(0, max_loc.x - DIR_MARGIN);
			valL[i] = max/maxL;

			mPtrLaneModel->candL.push_back(max_loc+ Point(0, max_loc.x - DIR_MARGIN));
			mPtrLaneModel->valL.push_back(valL[i]);
		}
		double maxP = -1;
		int mR, mL;
		const auto& lBINS_cm   = mLaneFilter->BINS_cm;

		for (int l = 0; l < CAND_AMT; l++)
		{
			for (int r = 0; r < CAND_AMT; r++)
			{

				int widthCm = (lBINS_cm( candR[r].x) - lBINS_cm(candL[l].x));
				if (widthCm < mLaneFilter->LANE.MIN_WIDTH) continue;
				if (widthCm > mLaneFilter->LANE.MAX_WIDTH) continue;


				int widthBase = candR[r].x - candL[l].x;
				int widthPur = candR[r].y - candL[l].y;

				double widthDif = widthPur - widthBase;

				double widthP = exp(-widthDif * widthDif / 60.0);

				double prob = valR[r] * valL[l] * widthP;

				if (prob > maxP)
				{
					maxP = prob;
					mR = r;
					mL = l;
				}
			}
		}

	//	cout << "prob = " << maxP << "\t" << bestWidthP << "\t" << maxL << "\t" << maxR << endl;

		int  lIdxBase_LB 	= candL[mL].x;
		int  lIdxBase_RB 	= candR[mR].x;


		mIdxPurview_RB = candR[mR].y;
		mIdxPurview_LB = candL[mL].y;


		sBufR[sBufIt] = candR[mR];
		sBufL[sBufIt] = candL[mL];

		int stable = 1;
		for (int i = 0; i < STAB_BUF_LEN; i++)
		{
			int a = abs(sBufR[sBufIt].x - sBufR[i].x);
			int c = abs(sBufL[sBufIt].x - sBufL[i].x);

			if (a > 2) stable = 0;
			if (c > 2) stable = 0;
		}

		stabilityCounter++;
		stabilityCounter *= stable;

		cout << (stable?"STABLE":"UNSTABLE") << "   " << stabilityCounter << endl;
		sBufIt = (sBufIt + 1)%STAB_BUF_LEN;


		if ((!tracking) && (stabilityCounter > 15)) tracking = 1;

		if (tracking)
		{











		}



	//Block Ends

#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Compute Weighted Histograms." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("COMPUTE_HISTOGRAMS")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("COMPUTE_HISTOGRAMS")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("COMPUTE_HISTOGRAMS")<<endl
				<<"******************************"<<endl<<endl;	
				#endif





#ifdef PROFILER_ENABLED
mProfiler.start("FILTERS_WAIT");
#endif 				
	mFuture.wait();

#ifdef PROFILER_ENABLED
 mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Waiting for worker thread to finish transition filters." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("FILTERS_WAIT")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("FILTERS_WAIT")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("FILTERS_WAIT")<<endl
				<<"******************************"<<endl<<endl;	
				#endif	




#ifdef PROFILER_ENABLED
mProfiler.start("SETUP_ASYNC_BUFFER_SHIFT");
#endif
	mFuture = std::async(std::launch::async, [this]
	{
	   WriteLock  lLock(_mutex, std::defer_lock);	

	   lLock.lock();
	   for ( std::size_t i = 0; i< mBufferPool->Probability.size()-1 ; i++ )
	   {
	 	mBufferPool->Probability[i+1].copyTo(mBufferPool->Probability[i]);		
		mBufferPool->GradientTangent[i+1].copyTo(mBufferPool->GradientTangent[i]);
	   }	
	   lLock.unlock();

	});

#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Setting up async task for shifting Buffers." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("SETUP_ASYNC_BUFFER_SHIFT")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("SETUP_ASYNC_BUFFER_SHIFT")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("SETUP_ASYNC_BUFFER_SHIFT")<<endl
				<<"******************************"<<endl<<endl;	
				#endif



#ifdef PROFILER_ENABLED
mProfiler.start("NORMALIZE_HISTOGRAM");
#endif
	{   
 /*  	    int64_t lSUM = 0;
	    //Normalising Base Histogram
	    lSUM = sum(mHistBase)[0] ;
	    mHistBase.convertTo(mHistBase_CV64F, CV_64F, SCALE_FILTER);
	    mHistBase_CV64F.convertTo(mHistBase, CV_32S, 1.0/lSUM );	 

	    //Normalising Purview Histogram
	    lSUM = sum(mHistPurview)[0];
	    mHistPurview.convertTo(mHistPurview_CV64F, CV_64F, SCALE_FILTER);
	    mHistPurview_CV64F.convertTo(mHistPurview, CV_32S, 1.0/lSUM );

	    //Normalising Far Histogram
	    lSUM = sum(mHistFar)[0];
	    mHistFar.convertTo(mHistFar_CV64F, CV_64F, SCALE_FILTER);
	    mHistFar_CV64F.convertTo(mHistFar, CV_32S, 1.0/lSUM );*/
	}
	
#ifdef PROFILER_ENABLED
 mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Normalising Hitograms." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("NORMALIZE_HISTOGRAM")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("NORMALIZE_HISTOGRAM")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("NORMALIZE_HISTOGRAM")<<endl
				<<"******************************"<<endl<<endl;	
				#endif	



#ifdef PROFILER_ENABLED
mProfiler.start("HISTOGRAM_MATCHING");
#endif



//vector<BaseHistogramModel>& Models	= mLaneFilter->baseHistogramModels;

//mBaseHistModel = Models[2];























	//cerr << "Chosen NB prob: " << chosenProb << "\n";

#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Histogram Matching MAP Estimate LaneBoundaries." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("HISTOGRAM_MATCHING")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("HISTOGRAM_MATCHING")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("HISTOGRAM_MATCHING")<<endl
				<<"******************************"<<endl<<endl;	
				#endif




#ifdef DEBUG_FRAMES
{
	const int MARGIN_WIDTH = 300;
	cv::Mat FrameTest;
	FrameTest = cv::Mat::ones(FrameGRAY.rows, 2*MARGIN_WIDTH + FrameGRAY.cols, CV_8U) * 128;
	FrameGRAY.copyTo(FrameTest(cv::Rect(MARGIN_WIDTH, 0, FrameGRAY.cols, FrameGRAY.rows)));
	cv::cvtColor(FrameTest, FrameTest, cv::COLOR_GRAY2BGR);

	uint32_t mHistPurviewMax = 0;
	uint32_t mHistBaseMax = 0;
	for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
	{
		if (mHistPurview.at<uint32_t>(i) > mHistPurviewMax) mHistPurviewMax = mHistPurview.at<uint32_t>(i);
		if (mHistBase.at<uint32_t>(i)    > mHistBaseMax)    mHistBaseMax    = mHistBase.at<uint32_t>(i);
	}

	uint32_t BAR_MAX_HEIGHT = 200;
	uint32_t mHistPurviewScale = mHistPurviewMax / BAR_MAX_HEIGHT;
	uint32_t mHistBaseScale = mHistBaseMax / BAR_MAX_HEIGHT;

	for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
	{
		int x = mLaneFilter->PURVIEW_BINS.at<int32_t>(i, 0) + mLaneFilter->O_ICCS_ICS.x;
		int y = mLaneFilter->PURVIEW_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;
		int val = mHistPurview.at<uint32_t>(i) / mHistPurviewScale;
		if (val < 0) val = 1;
		cv::line(FrameTest, cvPoint(x+MARGIN_WIDTH, y), cvPoint(x+MARGIN_WIDTH, y - val), cvScalar(0, val, 0), 3);
	}

	for (size_t i = 0; i < mLaneFilter->COUNT_BINS; i++)
	{
		int x = mLaneFilter->BASE_BINS.at<int32_t>(i, 0) + mLaneFilter->O_ICCS_ICS.x;
		int y = mLaneFilter->BASE_LINE_ICCS + mLaneFilter->O_ICCS_ICS.y;
		int val = mHistBase.at<uint32_t>(i) / mHistBaseScale;
		if (val < 0) val = 1;
		cv::line(FrameTest, cvPoint(x+MARGIN_WIDTH, y), cvPoint(x+MARGIN_WIDTH, y - val), cvScalar(val, 0, 0), 5);
	}

	cv::imshow("TrackingLaneDAG", FrameTest);
}
#endif
#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE) <<endl
				<<"******************************"<<endl
				<<  "Histogram Matching MAP Estimate VanishingPt."<<endl
				<<  "Max Time: " << mProfiler.getMaxTime("VP_HISTOGRAM_MATCHING")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("VP_HISTOGRAM_MATCHING")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("VP_HISTOGRAM_MATCHING")<<endl
		  		<<"******************************"<<endl<<endl;	
				#endif


//int  lIdxBase_LB 	= mBaseHistModel.binIdxBoundary_left;
//int  lIdxBase_RB 	= mBaseHistModel.binIdxBoundary_right;

/*
mIdxPurview_RB = lIdxBase_RB;
mIdxPurview_LB = lIdxBase_LB;

if (trackingCnt++ > 10)
{
	cout << "Tracking!\n";
	cv::Point hL(lastPosL.x, mHistBase.rows/2 + lastPosL.y - lastPosL.x);
	cv::Point hR(lastPosR.x, mHistBase.rows/2 + lastPosR.y - lastPosR.x);

	cout << hL << hR << endl;
	int M = 4;
	double probMax = -1;

	for (int dLx = -M; dLx <= M; dLx++)
		for (int dLy = -M; dLy <= M; dLy++)
			for (int dRx = -M; dRx <= M; dRx++)
				for (int dRy = -M; dRy <= M; dRy++)
				{
					cv::Point pL(hL+cv::Point(dLx, dLy));
					cv::Point pR(hR+cv::Point(dRx, dRy));

					if (pL.x < 0) continue;
					if (pR.x < 0) continue;
					if (pR.x >= mHistBase.rows) continue;
					if (pR.y >= mHistBase.rows) continue;


					double histR = hist2dr.at<int32_t>(pR.y, pR.x);
					double histL = hist2dl.at<int32_t>(pL.y, pL.x);
					double pLpos = 1;
					double pRpos = 1;
					if (dLx < 0) pLpos = exp(-dLx * dLx / 100.0 );
					if (dRx > 0) pRpos = exp(-dRx * dRx / 100.0 );
					double dW = dLy - dRy;
					double pWidth = exp(-dW * dW / 100.0 );

					double prob = histR * histL * pLpos * pRpos * pWidth;

					if (prob > probMax)
					{
						probMax = prob;
						lIdxBase_LB = pL.x;
						lIdxBase_RB = pR.x;
						mIdxPurview_RB = pR.y + pR.x - mHistBase.rows/2;
						mIdxPurview_LB = pL.y + pL.x - mHistBase.rows/2;
					}

				}
}



*/

#ifdef PROFILER_ENABLED
mProfiler.start("ASSIGN_LANE_MODEL");
#endif
	{
	   const auto& lBINS_cm   = mLaneFilter->BINS_cm;




	   const auto& lBASE_LB = mLaneFilter->BASE_BINS.at<int32_t>(lIdxBase_LB, 0);
	   const auto& lBASE_RB = mLaneFilter->BASE_BINS.at<int32_t>(lIdxBase_RB, 0);


	   const auto& lPURV_LB	  = mLaneFilter->PURVIEW_BINS.at<int32_t>(mIdxPurview_LB, 0);
	   const auto& lPURV_RB	  = mLaneFilter->PURVIEW_BINS.at<int32_t>(mIdxPurview_RB, 0);

	   float lLaneWidth	  = (lBINS_cm(mIdxPurview_RB) - lBINS_cm(mIdxPurview_LB));


	   lastPosL = cv::Point(lIdxBase_LB, mIdxPurview_LB);
	   lastPosR = cv::Point(lIdxBase_RB, mIdxPurview_RB);




	   //Set LaneModel
	   mPtrLaneModel->boundaryLeft[0] 	  = lBASE_LB;
	   mPtrLaneModel->boundaryLeft[1] 	  = lPURV_LB;

	   mPtrLaneModel->boundaryRight[0] 	  = lBASE_RB;
	   mPtrLaneModel->boundaryRight[1] 	  = lPURV_RB;

	   mPtrLaneModel->boundaryLeft_cm[0]  	  = lBINS_cm(lIdxBase_LB);
	   mPtrLaneModel->boundaryLeft_cm[1]  	  = lBINS_cm(mIdxPurview_LB);

	   mPtrLaneModel->boundaryRight_cm[0] 	  = lBINS_cm(lIdxBase_RB);
	   mPtrLaneModel->boundaryRight_cm[1] 	  = lBINS_cm(mIdxPurview_RB);

	   mPtrLaneModel->width_cm		  = lLaneWidth;
	   mPtrLaneModel->vanishingPt	  	  = mVanishPt;
	}
#ifdef PROFILER_ENABLED
mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE) <<endl
				<<"******************************"<<endl
				<<  "Assigning Lane Model."<<endl
				<<  "Max Time: " << mProfiler.getMaxTime("ASSIGN_LANE_MODEL")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("ASSIGN_LANE_MODEL")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("ASSIGN_LANE_MODEL")<<endl
		  		<<"******************************"<<endl<<endl;	
				#endif




#ifdef PROFILER_ENABLED
mProfiler.start("BUFFER_SHIFT_WAIT");
#endif 				
	mFuture.wait();

#ifdef PROFILER_ENABLED
 mProfiler.end();
LOG_INFO_(LDTLog::TIMING_PROFILE)<<endl
				<<"******************************"<<endl
				<<  "Waiting for async task to finish shifting buffers." <<endl
				<<  "Max Time: " << mProfiler.getMaxTime("BUFFER_SHIFT_WAIT")<<endl
				<<  "Avg Time: " << mProfiler.getAvgTime("BUFFER_SHIFT_WAIT")<<endl
				<<  "Min Time: " << mProfiler.getMinTime("BUFFER_SHIFT_WAIT")<<endl
				<<"******************************"<<endl<<endl;	
				#endif	

}//extractLanes

