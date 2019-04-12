/*
 * JsonOutput.h
 *
 *  Created on: Apr 12, 2019
 *      Author: msz
 */

#ifndef TUELDT_JSONOUTPUT_H_
#define TUELDT_JSONOUTPUT_H_

#include <string>
#include <vector>
#include "LaneModel.h"

using namespace std;
using namespace cv;

class JsonOutput {
public:
	JsonOutput();
	virtual ~JsonOutput();
	void print(const cv::UMat& FRAME, const LaneModel& Lane, string fileName);

private:
	vector<int> mLines;

};

#endif /* TUELDT_JSONOUTPUT_H_ */
