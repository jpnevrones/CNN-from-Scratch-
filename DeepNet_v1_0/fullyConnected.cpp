/*	fullyConnected.h : Header file - fullyConnected Network layer
*	version 0.1.2
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft-- 04 / 04 / 2017 --JP v0.1.1
*		-Fully Connected layer implemenataion-- 04 / 04 / 2017 --JP v0.1.1
*		-Fully Connected base class template -- 04 / 04 / 2017 --JP v0.1.1
*		-Fully Connected Feedforward backprop and update-- 04 / 11 / 2017 --JP v0.1.2
*
*/

#include "layer.h"

fullyConnected::fullyConnected() {}
fullyConnected::~fullyConnected() {

}

void fullyConnected::init(string namelayer, int hiddensize, double weightdecay, string resformat) {
	layerType = "fullyConnected";
	layerName = namelayer;
	resFormat = resformat;
	size = hiddensize;
	weightDecay = weightdecay;
}

void fullyConnected::initWeight(layer* prevLayer) {

	int inputsize = 0;
	if (prevLayer->resFormat == "image") {
		inputsize = prevLayer->resVector[0].size() * prevLayer->resVector[0][0].rows * prevLayer->resVector[0][0].cols * 3;
	}
	else {
		inputsize = prevLayer->resMatrix.rows;
	}
	double epsilon = 0.12;
	weight = Mat::ones(size, inputsize, CV_64FC1);
	randu(weight, Scalar(-1.0), Scalar(1.0));
	weight = weight * epsilon;
	bias = Mat::zeros(size, 1, CV_64FC1);
	weightGradient = Mat::zeros(weight.size(), CV_64FC1);
	biasGradient = Mat::zeros(bias.size(), CV_64FC1);
	weightD2 = Mat::zeros(weight.size(), CV_64FC1);
	biasD2 = Mat::zeros(bias.size(), CV_64FC1);

	// updater
	velocityWeight = Mat::zeros(weight.size(), CV_64FC1);
	velocityBias = Mat::zeros(bias.size(), CV_64FC1);
	weightSecD = Mat::zeros(weight.size(), CV_64FC1);
	biasSecD = Mat::zeros(bias.size(), CV_64FC1);
	iter = 0;
	mu = 1e-2;
	fullyConnected::setMomentum();
}

void fullyConnected::setMomentum() {
	if (iter < 30) {
		momtumdbydx = momtumWeightInit;
		momtumSecD = momtumSecDevInit;
	}
	else {
		momtumdbydx = momtumWeightAdj;
		momtumSecD = momtumSecDevAdj;
	}
}

void fullyConnected::update(int numOfIteration) {
	iter = numOfIteration;
	if (iter == 30) fullyConnected::setMomentum();
	weightSecD = momtumSecD * weightSecD + (1.0 - momtumSecD) * weightD2;
	learning_rate = lrateweight / (weightSecD + mu);
	velocityWeight = velocityWeight * momtumdbydx + (1.0 - momtumdbydx) * weightGradient.mul(learning_rate);
	weight -= velocityWeight;

	biasSecD = momtumSecD * biasSecD + (1.0 - momtumSecD) * biasD2;
	learning_rate = lratebias / (biasSecD + mu);
	velocityBias = velocityBias * momtumdbydx + (1.0 - momtumdbydx) * biasGradient.mul(learning_rate);
	bias -= velocityBias;
}

void fullyConnected::feedForward(int numOfSamples, layer* prevLayer, bool train) {
	//Feedforward is same for test and train pahse, bool paarmeter is added to keep all feedforward layer function to have common syntax
	Mat input;
	if (prevLayer->resFormat == "image") {
		convert(prevLayer->resVector, input);
	}
	else {
		prevLayer->resMatrix.copyTo(input);
	}
	Mat tempActivation = weight * input + repeat(bias, 1, numOfSamples);
	tempActivation.copyTo(resMatrix);
}

void fullyConnected::backprop(int numOfSamples, layer* prevLayer, layer* nextLayer) {

	Mat input;
	if (prevLayer->resFormat == "image") {
		convert(prevLayer->resVector, input);
	}
	else {
		prevLayer->resMatrix.copyTo(input);
	}

	if (nextLayer->resFormat == "image") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
	}
	else {
		Mat derivative;
		Mat secDerivative;
		nextLayer->deltaMatrix.copyTo(derivative);
		nextLayer->secDmatrix.copyTo(secDerivative);

		weightGradient = derivative * input.t() / numOfSamples + weightDecay * weight;
		biasGradient = reduceMat(derivative, 1, CV_REDUCE_SUM) / numOfSamples;
		weightD2 = secDerivative * powMat(input.t(), 2.0) / numOfSamples + weightDecay;
		biasD2 = reduceMat(secDerivative, 1, CV_REDUCE_SUM) / numOfSamples;

		Mat tmp = weight.t() * derivative;
		tmp.copyTo(deltaMatrix);
		tmp = powMat(weight.t(), 2.0) * secDerivative;
		tmp.copyTo(secDmatrix);
	}
}


























