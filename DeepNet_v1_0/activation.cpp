/*  activation.h : Header file - activation function support for the network
*	version 0.1.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/14/2017 --    JP v0.1.1
*		- activation base class template -- 04/14/2017 --    JP v0.1.2
*		- activation Feedforward -- 04/15/2017 --    JP v0.1.3
*
*/
#include "layer.h"


activation::activation() {}
activation::~activation() {
}

Mat nonLinear(const Mat &, int);
Mat dbydxnonLinear(const Mat &, int);

void activation::init(string namelayer, int pmethod, string resformat) {
	layerType = "activation";
	layerName = namelayer;
	method = pmethod;
	resFormat = resformat;
}

void activation::feedForward(int nsamples, layer* prevLayer, bool train) {
	//Feedforward is same for test and train pahse, bool paarmeter is added to keep all feedforward layer function to have common syntax
	if (resFormat == "matrix") {
		Mat input;
		if (prevLayer->resFormat == "matrix") {
			prevLayer->resMatrix.copyTo(input);
		}
		else {
			convert(prevLayer->resVector, input);
		}
		Mat res = nonLinear(input, method);
		res.copyTo(resMatrix);
	}
	else { // resFormat == "image"

		std::vector<std::vector<Mat> > input;
		if (prevLayer->resFormat == "image") {
			input = prevLayer->resVector;
		}
		else {
			cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
			return;
		}

		resVector.clear();
		resVector.resize(input.size());
		for (int i = 0; i < resVector.size(); i++) {
			resVector[i].resize(input[i].size());
		}
		for (int i = 0; i < input.size(); i++) {
			for (int j = 0; j < input[i].size(); j++) {
				Mat res = parallel(nonLinear, input[i][j], method);
				res.copyTo(resVector[i][j]);
			}
		}
		input.clear();
		std::vector<std::vector<Mat> >().swap(input);
	}
}


void activation::backprop(int nsamples, layer* prevLayer, layer* nextLayer) {

	if (resFormat == "matrix") {

		Mat derivative;
		Mat secDerivative;
		if (nextLayer->resFormat == "matrix") {
			nextLayer->deltaMatrix.copyTo(derivative);
			nextLayer->secDmatrix.copyTo(secDerivative);
		}
		else {
			convert(nextLayer->deltaVector, derivative);
			convert(nextLayer->secDvector, secDerivative);
		}
		Mat input;
		if (prevLayer->resFormat == "image") {
			convert(prevLayer->resVector, input);
		}
		else {
			prevLayer->resMatrix.copyTo(input);
		}
		Mat tmp = dbydxnonLinear(input, method);
		Mat tmp2 = derivative.mul(tmp);
		tmp2.copyTo(deltaMatrix);

		tmp2 = secDerivative.mul(powMat(tmp, 2.0));
		tmp2.copyTo(secDmatrix);
	}
	else {
		if (prevLayer->resFormat != "image") {
			cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
			return;
		}
		std::vector<std::vector<Mat> > derivative;
		std::vector<std::vector<Mat> > secDerivative;
		std::vector<std::vector<Mat> > input(prevLayer->resVector);

		if (nextLayer->resFormat == "matrix") {
			convert(nextLayer->deltaMatrix, derivative, nsamples, resVector[0][0].rows);
			convert(nextLayer->secDmatrix, secDerivative, nsamples, resVector[0][0].rows);
		}
		else {
			derivative = nextLayer->deltaVector;
			secDerivative = nextLayer->secDvector;
		}

		deltaVector.clear();
		secDvector.clear();
		deltaVector.resize(derivative.size());
		secDvector.resize(derivative.size());
		for (int i = 0; i < deltaVector.size(); i++) {
			deltaVector[i].resize(derivative[i].size());
			secDvector[i].resize(derivative[i].size());
		}
		for (int i = 0; i < derivative.size(); i++) {
			for (int j = 0; j < derivative[i].size(); j++) {
				Mat res = parallel(dbydxnonLinear, input[i][j], method);
				Mat tmp = derivative[i][j].mul(res);
				tmp.copyTo(deltaVector[i][j]);
				tmp = secDerivative[i][j].mul(powMat(res, 2.0));
				tmp.copyTo(secDvector[i][j]);
			}
		}
		derivative.clear();
		std::vector<std::vector<Mat> >().swap(derivative);
		secDerivative.clear();
		std::vector<std::vector<Mat> >().swap(secDerivative);
		input.clear();
		std::vector<std::vector<Mat> >().swap(input);

	}
}

Mat nonLinear(const Mat &matrix, int method) {
	if (method == RELU) {
		return relu(matrix);
	}
	else if (method == TANH) {
		return tanh(matrix);
	}
	else if (method == LEAKYRELU) {
		return leakyRelu(matrix);
	}
	else {
		return sigmoid(matrix);
	}
}

Mat dbydxnonLinear(const Mat &matrix, int method) {
	if (method == RELU) {
		return dbydxRelu(matrix);
	}
	else if (method == TANH) {
		return dbydxTanh(matrix);
	}
	else if (method == LEAKYRELU) {
		return dbydxLeakyReLU(matrix);
	}
	else {
		return dbydxSigmoid(matrix);
	}
}

