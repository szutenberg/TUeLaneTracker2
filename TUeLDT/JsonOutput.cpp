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


void JsonOutput::print(const cv::UMat& FRAME, const LaneModel& Lane, string fileName)
{
	Point mO_ICCS_ICS(FRAME.cols/2, FRAME.rows/2);
	int mBASE_LINE_ICS = 480; // TODO fix it

	vector<Point> lBoundaryPts_L;
	vector<Point> lBoundaryPts_R;
	vector<Point> lBoundaryPts_M;

	//Transform VP to Image Coordinate System
	int VP_V =  Lane.vanishingPt.V + mO_ICCS_ICS.y;
	int VP_H =  Lane.vanishingPt.H + mO_ICCS_ICS.x;

	//Lane Bundaries
	lBoundaryPts_L.push_back( Point( Lane.boundaryLeft[0]  + mO_ICCS_ICS.x, mBASE_LINE_ICS) );
	lBoundaryPts_R.push_back( Point( Lane.boundaryRight[0] + mO_ICCS_ICS.x, mBASE_LINE_ICS) );
	lBoundaryPts_M.push_back( (lBoundaryPts_L[0] + lBoundaryPts_R[0])/2.0 );

	float lSlopeLeft =  (float)( VP_V - 	mBASE_LINE_ICS ) /(VP_H - lBoundaryPts_L[0].x);
	float lSlopeRight = (float)( VP_V -	mBASE_LINE_ICS ) /(VP_H - lBoundaryPts_R[0].x);

	lBoundaryPts_L.push_back(lBoundaryPts_L[0]);
	lBoundaryPts_L[1].x  += 	-round((mBASE_LINE_ICS) / lSlopeLeft);
	lBoundaryPts_L[1].y 	+= 	-round((mBASE_LINE_ICS));

	lBoundaryPts_R.push_back(lBoundaryPts_R[0]);
	lBoundaryPts_R[1].x  += 	-round((mBASE_LINE_ICS) / lSlopeRight);
	lBoundaryPts_R[1].y 	+= 	-round((mBASE_LINE_ICS));

	lBoundaryPts_M.push_back( (lBoundaryPts_L[1] + lBoundaryPts_R[1])/2.0);

	printf("{\"raw_file\": \"%s\", \"lanes\": [[", fileName.c_str());
	float dx = (float)(lBoundaryPts_L[0].x - lBoundaryPts_L[1].x) / (lBoundaryPts_L[0].y - lBoundaryPts_L[1].y);
	bool firstIt = true;
	for (int line : mLines)
	{
		if (!firstIt) printf(", ");
		float x = lBoundaryPts_L[0].x + dx * (line - lBoundaryPts_L[0].y);
		printf("%.1f", x);
		firstIt = false;
	}
	printf("], [");

	dx = (float)(lBoundaryPts_R[0].x - lBoundaryPts_R[1].x) / (lBoundaryPts_R[0].y - lBoundaryPts_R[1].y);
	firstIt = true;
	for (int line : mLines)
	{
		if (!firstIt) printf(", ");
		float x = lBoundaryPts_R[0].x + dx * (line - lBoundaryPts_R[0].y);
		printf("%.1f", x);
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

