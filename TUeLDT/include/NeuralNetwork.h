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
	static cv::Mat getResult();

	virtual ~NeuralNetwork();
private:
	static void threadFunction();
	std::thread mThread;
	int mWidth;
	int mHeight;
};

#endif /* TUELDT_NEURALNETWORK_H_ */
