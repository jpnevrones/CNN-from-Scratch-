/*	softmax.h : Header file - softmax Network layer 
*	version 0.1.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/04/2017 --    JP v0.1.1
*		- softmax layer implemenataion -- 04/13/2017 --    JP v0.1.1
*		- softmax base class template -- 04/05/2017 --    JP v0.1.1
*		- softmax Feedforward backprop and update -- 04/12/2017 --    JP v0.1.2
*		- Update to softmax Feedforward backprop and update function 
*		  Correction to bug with derivatives		-- 04/13/2017 --    JP v0.1.3
*
*/
#include "layer.h"

softmax::softmax() {}
softmax::~softmax() {
}

void softmax::init(string namelayer, int numclasses, double weightdecay, string resformat) {
	layerType = "softmax";
	layerName = namelayer;
	resFormat = resformat;
	resSize = numclasses;
	weightDecay = weightdecay;
}

void softmax::initWeight(layer* prevLayer) {

	int inputsize = 0;
	if (prevLayer->resFormat == "image") {
		inputsize = prevLayer->resVector[0].size() * prevLayer->resVector[0][0].rows * prevLayer->resVector[0][0].cols * 3;
	}
	else {
		inputsize = prevLayer->resMatrix.rows;
	}
	double epsilon = 0.12;
	weight = Mat::ones(resSize, inputsize, CV_64FC1);
	randu(weight, Scalar(-1.0), Scalar(1.0));
	weight = weight * epsilon;
	bias = Mat::zeros(resSize, 1, CV_64FC1);
	weightGradient = Mat::zeros(weight.size(), CV_64FC1);
	biasGradient = Mat::zeros(bias.size(), CV_64FC1);
	weightD2 = Mat::zeros(weight.size(), CV_64FC1);
	biasD2 = Mat::zeros(bias.size(), CV_64FC1);


	velocityWeight = Mat::zeros(weight.size(), CV_64FC1);
	velocityBias = Mat::zeros(bias.size(), CV_64FC1);
	weightSecD = Mat::zeros(weight.size(), CV_64FC1);
	biasSecD = Mat::zeros(bias.size(), CV_64FC1);
	iter = 0;
	mu = 1e-2;
	softmax::setMomentum();
}

void softmax::setMomentum() {
	if (iter < 30) {
		momtumdbydx = momtumWeightInit;
		momtumSecD = momtumSecDevInit;
	}
	else {
		momtumdbydx = momtumWeightAdj;
		momtumSecD = momtumSecDevAdj;
	}
}

void softmax::update(int numOfIteration) {
	iter = numOfIteration;
	if (iter == 30) softmax::setMomentum();
	weightSecD = momtumSecD * weightSecD + (1.0 - momtumSecD) * weightD2;
	learning_rate = lrateweight / (weightSecD + mu);
	velocityWeight = velocityWeight * momtumdbydx + (1.0 - momtumdbydx) * weightGradient.mul(learning_rate);
	weight -= velocityWeight;

	biasSecD = momtumSecD * biasSecD + (1.0 - momtumSecD) * biasD2;
	learning_rate = lratebias / (biasSecD + mu);
	velocityBias = velocityBias * momtumdbydx + (1.0 - momtumdbydx) * biasGradient.mul(learning_rate);
	bias -= velocityBias;
}

void softmax::feedForward(int numOfSamples, layer* prevLayer, bool train) {
	Mat input;
	if (prevLayer->resFormat == "image") {
		convert(prevLayer->resVector, input);
	}
	else {
		prevLayer->resMatrix.copyTo(input);
	}
	Mat matrix = weight * input + repeat(bias, 1, numOfSamples);
	if (train) 
	{
		matrix -= repeat(reduceMat(matrix, 0, CV_REDUCE_MAX), matrix.rows, 1);
		matrix = expMat(matrix);
		Mat res = divMat(matrix, repeat(reduceMat(matrix, 0, CV_REDUCE_SUM), matrix.rows, 1));
		res.copyTo(resMatrix);
	}
	else
	{
		matrix.copyTo(resMatrix);
	}

}

void softmax::backprop(int numOfSamples, layer* prevLayer, Mat& groundTruth) {

	Mat input;
	if (prevLayer->resFormat == "image") {
		convert(prevLayer->resVector, input);
	}
	else {
		prevLayer->resMatrix.copyTo(input);
	}
	Mat derivative = groundTruth - resMatrix;
	weightGradient = -derivative * input.t() / numOfSamples + weightDecay * weight;
	biasGradient = -reduceMat(derivative, 1, CV_REDUCE_SUM) / numOfSamples;
	weightD2 = powMat(derivative, 2.0) * powMat(input.t(), 2.0) / numOfSamples + weightDecay;
	biasD2 = reduceMat(powMat(derivative, 2.0), 1, CV_REDUCE_SUM) / numOfSamples;

	Mat tmp = -weight.t() * derivative;
	tmp.copyTo(deltaMatrix);
	tmp = powMat(weight.t(), 2.0) * powMat(derivative, 2.0);
	tmp.copyTo(secDmatrix);

}

























