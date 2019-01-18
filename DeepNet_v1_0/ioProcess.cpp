/*	ioProcess.cpp : source file - contains the matrix based compuation utilities
*	and other set of utilities used by the core Framework
*	version 0.1.2
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 03/31/2017 --    JP v0.1.1
*		- Matrix based Computation utilities -- 03/31/2017 -- JP v0.1.1
*		- OpenCV required for these functions -- 04/04/2017 -- JP v0.1.2
*				- TODO - more to be added as required
*
*/


#include "DeepNet_v1_0.h"
#include <direct.h>

using namespace cv;
using namespace std; 

void getBatch(string fileName, vector<Mat> &vec, Mat &label) 
{
	ifstream file(fileName, ios::binary);
	if (file.is_open()) 
	{
		int numOfImages = 10000;
		int width = 32;
		int height = 32;
		for (int i = 0; i < numOfImages; ++i) 
		{
			unsigned char tmpLabel = 0;

			file.read((char*)&tmpLabel, sizeof(tmpLabel));
			vector<Mat> channels(3);
			Mat image = Mat::zeros(width, height, CV_8UC3);
			for (int ch = 0; ch < 3; ++ch) 
			{
				Mat tmpImage = Mat::zeros(width, height, CV_8UC1);
				for (int imgWidth = 0; imgWidth < width; ++imgWidth) {
					for (int imgHeight = 0; imgHeight < height; ++imgHeight) {
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						tmpImage.at<uchar>(imgWidth, imgHeight) = (int)temp;
					}
				}
				tmpImage.copyTo(channels[ch]);
				//                channels.push_back(tmpImage);
			}
			merge(channels, image);
			vec.push_back(image);
			label.ATD(0, i) = (double)tmpLabel;
		}
	}
	file.close();
}

void getCIFAR10(vector<Mat> &trainX, Mat &trainY, vector<Mat> &testX,  Mat &testY) 
{
	string fileName;
	fileName = "cifar-10-batches-bin/data_batch_";
	vector<Mat> labels;
	vector<vector<Mat> > batches;
	int numOfBatch = 5;
	//get train batch file
	for (int i = 1; i <= numOfBatch; i++) {
		vector<Mat> tmpBatch;
		Mat tmpLabel = Mat::zeros(1, 10000, CV_64FC1);
		string name = fileName + std::to_string((long long)i) + ".bin";
		getBatch(name, tmpBatch, tmpLabel);
		labels.push_back(tmpLabel);
		batches.push_back(tmpBatch);
		tmpBatch.clear();
	}
	// trainX
	trainX.reserve(batches[0].size() * numOfBatch);
	for (int i = 0; i < numOfBatch; i++) {
		trainX.insert(trainX.end(), batches[i].begin(), batches[i].end());
	}
	// trainY
	trainY = Mat::zeros(1, 10000 * numOfBatch, CV_64FC1);
	Rect roi;
	Mat subView;
	for (int i = 0; i < numOfBatch; i++) {
		roi = cv::Rect(labels[i].cols * i, 0, labels[i].cols, 1);
		subView = trainY(roi);
		labels[i].copyTo(subView);
	}
	// get test batch file
	fileName = "cifar-10-batches-bin/test_batch.bin";
	testY = Mat::zeros(1, 10000, CV_64FC1);

	getBatch(fileName, testX, testY);
	preProcessing(trainX, testX);

	cout << "Total number of training sample :  " << trainX.size() << " images, images dimension :  " << trainX[0].rows 
		<< " X " << trainX[0].cols << endl;
	cout << "Total number of training sample :  " << testX.size() << " images, images dimension :  " << testX[0].rows 
		<< " X " << testX[0].cols  << endl;
	cout << "There are " << trainY.cols << " training labels and " << testY.cols << " testing labels." 
		<< endl << endl;
}


Mat concat(const vector<Mat> &vec) {
	int height = vec[0].rows * vec[0].cols;
	int width = vec.size();
	Mat res = Mat::zeros(height, width, CV_64FC3);
	for (int i = 0; i < vec.size(); i++) {
		Rect roi = Rect(i, 0, 1, height);
		Mat subView = res(roi);
		Mat ptmat = vec[i].reshape(0, height);
		ptmat.copyTo(subView);
	}
	return res;
}


void preProcessing(vector<Mat> &trainX, vector<Mat> &testX) {
	for (int i = 0; i < trainX.size(); i++) {
		//cvtColor(trainX[i], trainX[i], CV_RGB2YCrCb);
		trainX[i].convertTo(trainX[i], CV_64FC3, 1.0 / 255, 0);
	}
	for (int i = 0; i < testX.size(); i++) {
		//cvtColor(testX[i], testX[i], CV_RGB2YCrCb);
		testX[i].convertTo(testX[i], CV_64FC3, 1.0 / 255, 0);
	}

	// first convert vec of mat into a single mat
	Mat tmp = concat(trainX);
	Mat tmp2 = concat(testX);
	Mat alldata = Mat::zeros(tmp.rows, tmp.cols + tmp2.cols, CV_64FC3);

	tmp.copyTo(alldata(Rect(0, 0, tmp.cols, tmp.rows)));
	tmp2.copyTo(alldata(Rect(tmp.cols, 0, tmp2.cols, tmp.rows)));

	Scalar mean;
	Scalar stddev;
	meanStdDev(alldata, mean, stddev);

	for (int i = 0; i < trainX.size(); i++) {
		divide(trainX[i] - mean, stddev, trainX[i]);
	}
	for (int i = 0; i < testX.size(); i++) {
		divide(testX[i] - mean, stddev, testX[i]);
	}
}






/*void save2XML(std::vector<layer*> &, string path, string name) {

	_mkdir(path.c_str());
	string tmp = path + "/" + name + ".xml";
	FileStorage fs(tmp, FileStorage::WRITE);
	int conv = 0;
	int fc = 0;
	cout << "Saving DeepNet weights for the current Network Architecture" << endl;
	for (int i = 0; i < networkArch.size(); i++) {
		if (networkArch[i]->layerType == "convolutional") {
			tmp = "ConvolutionLayer_" + std::to_string(conv);
			for (int j = 0; j < ((convolution*)networkArch[i])->kernels.size(); j++) {
				string tmp2 = tmp + "_kernel_" + std::to_string(j);
				fs << (tmp2 + "_w") << ((convolution*)networkArch[i])->kernels[j]->weight;
			}
			++conv;
		}else if(networkArch[i] -> layerType == "fully_connected") {
			tmp = "FullyConnectedLayer_" + std::to_string(fc);
			fs << (tmp + "_w") << ((fullyConnected*)networkArch[i])->weight;
			++fc;
		}else if(networkArch[i] -> layerType == "softmax") {
			fs << ("SoftmaxLayer_") << ((softmax*)networkArch[i])->weight;
		}
	}
	fs.release();
	cout << "Save Completed Successfully" << endl;
	cout << "Saved to location : "<< tmp << endl;
}*/
