#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
	  std::unique_ptr<tensorflow::Session> session;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;




// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}



/*
 * NeuralNetwork.cpp
 *
 *  Created on: Jun 5, 2019
 *      Author: msz
 */

#include "NeuralNetwork.h"
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
using namespace std;
int NeuralNetwork::mPort = 0;

int initialized = 0; // FIXME

NeuralNetwork::NeuralNetwork(int port, int width, int height) {


	mWidth = width;
	mHeight = height;
	cerr << "Neural Network initialization!\n";
	mThread = std::thread(&threadFunction);
}


NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub

}
vector<Mat> mImgQueue;

void NeuralNetwork::processImage(cv::Mat img)
{
	while(mImgQueue.size() > 1);
	Mat wrzuta;
	img.copyTo(wrzuta);
	mImgQueue.push_back(wrzuta);
	return;

}

int resultReady = 0;
Mat result;
cv::Mat NeuralNetwork::getResult()
{
	if (!initialized) return cv::Mat(0,0,0);
	while(!resultReady);
	return result;
}


void NeuralNetwork::threadFunction()
{
	printf("Thread!");
	string graph = "./output_graph.pb";
	int mHeight = 260;
	int mWidth = 640;
	// First we load and initialize the model.
	Status load_graph_status = LoadGraph(graph, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return;
	}

	string input_layer = "input_3";
	string output_layer = "k2tfout_0";

	std::vector<Tensor> resized_tensors;

	tensorflow::Tensor input_tensor(tensorflow::DataType::DT_FLOAT,
			tensorflow::TensorShape({1, mHeight, mWidth, 3}));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>();

	initialized = 1;
	while(1)
	{

		Mat img;

		while(mImgQueue.empty());

		resultReady = 0;

		img = mImgQueue[0];
		mImgQueue.pop_back();

		int startY = img.rows - mHeight;
		cv::Rect lROI = cv::Rect(0, startY, mWidth, mHeight);
		cv::Mat imgROI;
		img(lROI).copyTo(imgROI);


		for (int y = 0; y < mHeight; y++) {
			for (int x = 0; x < mWidth; x++) {
				Vec3b pixel = imgROI.at<Vec3b>(y, x);

				input_tensor_mapped(0, y, x, 0) = pixel.val[2] / 128.0 - 1.0; //R
				input_tensor_mapped(0, y, x, 1) = pixel.val[1] / 128.0 - 1.0; //G
				input_tensor_mapped(0, y, x, 2) = pixel.val[0] / 128.0 - 1.0; //B
			}
		}

		const Tensor& resized_tensor = input_tensor;

		std::vector<Tensor> outputs;
		Status run_status = session->Run({{input_layer, resized_tensor}},
				{output_layer}, {}, &outputs);
		if (!run_status.ok()) {
			LOG(ERROR) << "Running model failed: " << run_status;
			return;
		}

		int outHeight = mHeight / 4;
		int outWidth = mWidth / 4;


		assert(outputs[0].dims() == 3);
		assert(outputs[0].dim_size(0) == outHeight);
		assert(outputs[0].dim_size(1) == outWidth);
		assert(outputs[0].dim_size(2) == 1);


		Mat outMat =  Mat::zeros(mHeight/4, mWidth/4, CV_32F);
		// tensor<float, 3>: 3 here because it's a 3-dimension tensor
		auto outputImg = outputs[0].tensor<float, 3>();

		for (int y = 0; y < outHeight; ++y) {
			for (int x = 0; x < outWidth; ++x)
			{
				float val = outputImg(y, x,0);
				outMat.at<float>(y,x) = val;
			}
		}

		resize(outMat, result, cv::Size(mWidth, mHeight), 0, 0, INTER_NEAREST);
		resultReady = 1;
	}
}
