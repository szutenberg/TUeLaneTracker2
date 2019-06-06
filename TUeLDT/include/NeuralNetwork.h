/*
 * NeuralNetwork.h
 *
 *  Created on: Jun 5, 2019
 *      Author: msz
 */

#ifndef TUELDT_NEURALNETWORK_H_
#define TUELDT_NEURALNETWORK_H_

#include "opencv2/opencv.hpp"
#include <thread>
#include <atomic>

using namespace cv;

class NeuralNetwork {
public:
	NeuralNetwork(int port, int width, int height);
	void processImage(cv::Mat img);
	cv::Mat getResult();
	cv::Mat getRecentResult();

	virtual ~NeuralNetwork();
	static int mPort;
private:
	static void threadFunction();
	std::thread mThread;

};

#endif /* TUELDT_NEURALNETWORK_H_ */
