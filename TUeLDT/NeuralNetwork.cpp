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

int sock = 0;

NeuralNetwork::NeuralNetwork(int port, int width, int height) {
	// TODO Auto-generated constructor stub
	mThread = std::thread(&threadFunction);
	NeuralNetwork::mPort = port;

}

NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub

	close(sock);
	sock = 0;
}

void NeuralNetwork::processImage(cv::Mat img)
{
	printf("Process %d %d\n", img.cols, img.rows);
	uchar* ptr = img.ptr<uchar>(0);
	send(sock , ptr , 640*480*3 , 0 );
	send(sock, ptr, 1, 0);
	std::this_thread::sleep_for(std::chrono::milliseconds(200));




}

cv::Mat NeuralNetwork::getResult()
{

	return cv::Mat(1,1, 1);

}

cv::Mat NeuralNetwork::getRecentResult()
{
return cv::Mat(1,1,1);

}

void NeuralNetwork::threadFunction()
{
	printf("Thread!");

//	while (1)
//	{
//		printf("alive\n");
//	    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//	}

	// Client side C/C++ program to demonstrate Socket programming

	int valread;
	struct sockaddr_in serv_addr;
	char *hello = "Hello from client";
	char buffer[1024] = {0};
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		cerr << "Socket creation error\n";
		return;
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(NeuralNetwork::mPort);

	// Convert IPv4 and IPv6 addresses from text to binary form
	if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
	{
		cerr << "Invalid address/ Address not supported\n";
		return;
	}

	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		cerr << "Connection Failed";
		return;
	}
	//send(sock , hello , strlen(hello) , 0 );
	printf("Hello message sent\n");
	//valread = read( sock , buffer, 1024);
	//printf("%s\n",buffer );

}
