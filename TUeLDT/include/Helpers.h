/*
 * Helpers.h
 *
 *  Created on: Mar 1, 2019
 *      Author: msz
 */

#ifndef TUELDT_HELPERS_H_
#define TUELDT_HELPERS_H_

#include<string>
#include<vector>
#include "opencv2/opencv.hpp"

using namespace std;

class Helpers {

public:
	static void sortFileNames(vector<cv::String>& vec);


private: // only static methods
	Helpers();
	virtual ~Helpers();
};

#endif /* TUELDT_HELPERS_H_ */
