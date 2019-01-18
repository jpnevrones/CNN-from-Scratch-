/*	dropout.h : Header file - dropout Network layer
*	version 0.1.2
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/04/2017 --    JP v0.1.1
*		- Dropout layer implemenataion -- 04/13/2017 --    JP v0.1.1
*		- Dropout base class template -- 04/04/2017 --    JP v0.1.1
*		- Dropout Feedforward backprop and update -- 04/12/2017 --    JP v0.1.2
*
*/
#include "layer.h"


dropout::dropout() {}
dropout::~dropout() {
}

void dropout::init(string namelayer, double dor, string resformat) {
	layerType = "dropout";
	layerName = namelayer;
	resFormat = resformat;
	dropoutRate = dor;
}

void dropout::feedForward(int numOfSamples, layer* prevLayer, bool train) {

	if (resFormat == "matrix") {
		Mat input;
		if (prevLayer->resFormat == "matrix") {
			prevLayer->resMatrix.copyTo(input);
		}
		else {
			convert(prevLayer->resVector, input);
		}
		Mat res;
		if (train)
		{	//train pahse
			res = getBernoulliMatrix(input.rows, input.cols, dropoutRate);
			res.copyTo(bernoulliMatrix);
			res = res.mul(input);
			
		}
		else		
		{
			//test phase
			input.copyTo(res);
			res = res.mul(dropoutRate);
		}
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
		bernoulliVector.clear();
		resVector.resize(input.size());
		bernoulliVector.resize(input.size());
		for (int i = 0; i < resVector.size(); i++) {
			resVector[i].resize(input[i].size());
			bernoulliVector[i].resize(input[i].size());
		}

		if (train)
		{	//train phase only
			for (int i = 0; i < input.size(); i++) {
				for (int j = 0; j < input[i].size(); j++) {
					vector<Mat> bnls(3);
					Mat bnl;
					for (int ch = 0; ch < 3; ch++) {
						Mat tmp = getBernoulliMatrix(input[i][j].rows, input[i][j].cols, dropoutRate);
						tmp.copyTo(bnls[ch]);
					}
					merge(bnls, bnl);
					Mat res;
					bnl.copyTo(bernoulliVector[i][j]);
					res = bnl.mul(input[i][j]);
					res.copyTo(resVector[i][j]);
				}
			}

		}
		else
		{	//test phase only
			for (int i = 0; i < input.size(); i++) {
				for (int j = 0; j < input[i].size(); j++) {
					Mat res;
					input[i][j].copyTo(res);
					res = res.mul(Scalar::all(dropoutRate));
					res.copyTo(resVector[i][j]);
				}
			}
		}

		input.clear();
		std::vector<std::vector<Mat> >().swap(input);
	}
}

void dropout::backprop(int numOfSamples, layer* prevLayer, layer* nextLayer) {

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
		Mat tmp = derivative.mul(bernoulliMatrix);
		Mat tmp2 = secDerivative.mul(powMat(bernoulliMatrix, 2.0));
		tmp.copyTo(deltaMatrix);
		tmp2.copyTo(secDmatrix);

	}
	else {
		if (prevLayer->resFormat != "image") {
			cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
			return;
		}
		std::vector<std::vector<Mat> > derivative;
		std::vector<std::vector<Mat> > secDerivative;
		if (nextLayer->resFormat == "matrix") {
			convert(nextLayer->deltaMatrix, derivative, numOfSamples, resVector[0][0].rows);
			convert(nextLayer->secDmatrix, secDerivative, numOfSamples, resVector[0][0].rows);
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

				Mat tmp = derivative[i][j].mul(bernoulliVector[i][j]);
				Mat tmp2 = secDerivative[i][j].mul(powMat(bernoulliVector[i][j], 2.0));
				tmp.copyTo(deltaVector[i][j]);
				tmp2.copyTo(secDvector[i][j]);
			}
		}
		derivative.clear();
		std::vector<std::vector<Mat> >().swap(derivative);
		secDerivative.clear();
		std::vector<std::vector<Mat> >().swap(secDerivative);
	}
}


