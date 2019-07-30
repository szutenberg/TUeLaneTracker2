/*
 * JsonOutput.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: msz
 */

#include "JsonOutput.h"


JsonOutput::JsonOutput() {
	// TODO Auto-generated constructor stub
	mLines.push_back(280); // TODO it will be taken from the config file
	mLines.push_back(340);
	mLines.push_back(400);
	mLines.push_back(440);
}


float JsonOutput::extractLine(int line, const vector<Point2f>* vec)
{
	for (size_t i = 0; i < vec->size()-1; i++)
	{
		if ( ((*vec)[i].y >= line) && (line > (*vec)[i+1].y) )
		{
			float dx = (float)((*vec)[i].x - (*vec)[i+1].x) / ((*vec)[i].y - (*vec)[i+1].y);
			return (*vec)[i].x + dx * (line - (*vec)[i].y);
		}
	}

	return 123456;
}

void JsonOutput::print(const cv::UMat& FRAME, const LaneModel& Lane, string fileName)
{
	Point mO_ICCS_ICS(FRAME.cols/2, FRAME.rows/2);
	int mBASE_LINE_ICS = 480; // TODO fix it
	int mPURVIEW_LINE_ICS = 480 - 120;
	vector<Point2f> lBoundaryPts_L; // for straight lines
	vector<Point2f> lBoundaryPts_R; // for straight lines
	const vector<Point2f>* boundaryL = &lBoundaryPts_L;
	const vector<Point2f>* boundaryR = &lBoundaryPts_R;

	if (!(Lane.curveL.empty() || Lane.curveR.empty()))
	{
		boundaryL = &Lane.curveL;
		boundaryR = &Lane.curveR;
	}


	//Transform VP to Image Coordinate System
	int VP_V =  Lane.vanishingPt.V + mO_ICCS_ICS.y;
	int VP_H =  Lane.vanishingPt.H + mO_ICCS_ICS.x;

	//Lane Bundaries
	lBoundaryPts_L.push_back( Point( Lane.boundaryLeft[0]  + mO_ICCS_ICS.x, mBASE_LINE_ICS) );
	lBoundaryPts_R.push_back( Point( Lane.boundaryRight[0] + mO_ICCS_ICS.x, mBASE_LINE_ICS) );

	float lSlopeLeft =  (float)( mPURVIEW_LINE_ICS - 	mBASE_LINE_ICS ) /(Lane.boundaryLeft[1] - Lane.boundaryLeft[0]);
	float lSlopeRight =  (float)( mPURVIEW_LINE_ICS - 	mBASE_LINE_ICS ) /(Lane.boundaryRight[1] - Lane.boundaryRight[0]);


	lBoundaryPts_L.push_back(lBoundaryPts_L[0]);
	lBoundaryPts_L[1].x  += 	-round((mBASE_LINE_ICS) / lSlopeLeft);
	lBoundaryPts_L[1].y 	+= 	-round((mBASE_LINE_ICS));

	lBoundaryPts_R.push_back(lBoundaryPts_R[0]);
	lBoundaryPts_R[1].x  += 	-round((mBASE_LINE_ICS) / lSlopeRight);
	lBoundaryPts_R[1].y 	+= 	-round((mBASE_LINE_ICS));


	printf("{\"raw_file\": \"%s\", \"lanes\": [[", fileName.c_str());
	bool firstIt = true;
	for (int line : mLines)
	{
		if (!firstIt) printf(", ");
		printf("%.1f", extractLine(line, boundaryL));
		firstIt = false;
	}
	printf("], [");

	firstIt = true;
	for (int line : mLines)
	{
		if (!firstIt) printf(", ");
		printf("%.1f", extractLine(line, boundaryR));
		firstIt = false;
	}

	printf("]], \"h_samples\": [");
	firstIt = true;
	for (int line : mLines)
	{
		if (!firstIt) printf(", ");
		printf("%d", line);
		firstIt = false;
	}
	printf("]}\n");
}

JsonOutput::~JsonOutput() {
	// TODO Auto-generated destructor stub

}

