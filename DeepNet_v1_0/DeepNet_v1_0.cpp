#include "pch.h"
#include "DeepNet_v1_0.h"


/*	deepNetMain : Defines the entry point for the console application.
*	version 0.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Initial draft by JP - Jithin Pradeep(jxp161430) on 03 / 31 / 2017 --v0.1
*	Last Update:
*/

#include <iostream>
using namespace cv;
using namespace std;
//Settings
bool isGradientTrue = false;
bool enableNetLog = false;
int trainingEpochs = 0;
int iterationPerEpoch = 0;

// Network Paremter
double momtumWeightInit = 0.5;
double momtumSecDevInit = 0.5;
double momtumWeightAdj = 0.95;
double momtumSecDevAdj = 0.90;
double lrateweight = 0.0;
double lratebias = 0.0;
std::vector<layer*> networkArch;
/*Supported layer function declartion with example for future reference
*	void inputLayer(string custLayerName, int batchSize);
*	void convolutionLayer(string custLayerName, int Numkernel, int convkernelSize, int featureMap,
*	int paddings, int convSteps, double weightDecay);
*	void fullyConnectedLayer(string custLayerName, int numOfHiddenUnits, double weightDecay);
*	void poolingFun(string custLayerName, string poolingMethod, int poolSteps, bool isoverlap, int windowsize);
*	void softmaxLayer(string custLayerName, int numOfOutputClass, double weightDecay);
*	void normalize(string custLayerName, double alpha, double beta, double k, double n);
*	void activationFun(string custLayerName, string ActivationMethod);
*	void dropoutLayer(string custLayerName, double dropuoutRate);
*/

void deepNetArchitecture()
{



	inputLayer(networkArch, "INPUT", 100);

	convolutionLayer(networkArch, "CONVOLUTION1", 16, 7, 0, 0, 1, 1e-6);
	activationFun(networkArch, "ACTIVATION1", "LRELU");
	normalize(networkArch, "NORM1", 0.000125, 0.75, 2.0, 5);
	poolingFun(networkArch, "POOLING1", "MAXPOOL", 2, false, 0);

	convolutionLayer(networkArch, "CONVOLUTION2", 16, 8, 0, 0, 1, 1e-6);
	activationFun(networkArch, "ACTIVATION2", "LRELU");
	normalize(networkArch, "NORM2", 0.000125, 0.75, 2.0, 5);
	poolingFun(networkArch, "POOLING2", "MAXPOOL", 2, false, 0);

	fullyConnectedLayer(networkArch, "FULLYCONNECTED3", 512, 1e-6);
	activationFun(networkArch, "ACTIVATION3", "LRELU");
	dropoutLayer(networkArch, "DROPOUT3", 0.5);

	fullyConnectedLayer(networkArch, "FULLYCONNECTED4", 128, 1e-6);
	activationFun(networkArch, "ACTIVATION4", "LRELU");
	dropoutLayer(networkArch, "DROPOUT4", 0.5);

	softmaxLayer(networkArch, "OUTPUT", 10, 1e-6);

}

void deepNetMainFunc()
{
	std::vector<Mat> trainX;
	std::vector<Mat> testX;
	Mat trainY, testY;
	cout << "Getting CIFAR10 dataset " << endl;
	getCIFAR10(trainX, trainY, testX, testY);

	deepNetArchitecture();
	cout << "Intializing DeepNet " << endl;
	trainDeepNet(trainX, trainY, testX, testY, networkArch);

	trainX.clear();
	std::vector<Mat>().swap(trainX);
	testX.clear();
	std::vector<Mat>().swap(testX);
}



int main()
{

	deepNetMainFunc();
	return 0;
}

