/*
 * Helpers.cpp
 *
 *  Created on: Mar 1, 2019
 *      Author: msz
 */

#include "Helpers.h"

Helpers::Helpers() {
	// TODO Auto-generated constructor stub

}


// TODO - put into class
struct sortingHelper{
	int pos;
	int value;
};

bool operator<(const sortingHelper a, const sortingHelper b) { return a.value < b.value; }

void Helpers::sortFileNames(vector<cv::String>& vec)
{
	sortingHelper tmp;
	vector<sortingHelper> sorter;

	int i = 0;
	for (cv::String str : vec)
	{
		tmp.pos = i++;
		tmp.value = 0;
		const char *cstr = str.c_str();
		size_t it = str.rfind('/');
		if (it == cv::String::npos) continue;

		it++;
		while((cstr[it] >= '0') && (cstr[it] <= '9'))
		{
			int digit = cstr[it] - '0';
			tmp.value = tmp.value * 10 + digit;
			it++;
		}
		sorter.push_back(tmp);
	}

	vector<cv::String> inVec;

	inVec = vec;
	vec.clear();

	sort(sorter.begin(), sorter.end());

	for (sortingHelper tmp : sorter)
	{
		vec.push_back(inVec[tmp.pos]);
	}
}


Helpers::~Helpers() {
	// TODO Auto-generated destructor stub
}

