/*	pooling.h : Header file - pooling Network layer
*	version 0.1.2
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/04/2017 --    JP v0.1.1
*		- Pooling layer implemenataion -- 04/13/2017 --    JP v0.1.1
*		- Pooling base class template -- 04/05/2017 --    JP v0.1.2
*		- Pooling Feedforward backprop and update -- 04/12/2017 --    JP v0.1.3
*		- Update to Pooling Feedforward backprop and update function
*		  to support overlap while pooling and reverse pooling task -- 04/12/2017 --    JP v0.1.4		
*		- Correction to bug during initilization while overlap is supported -- 04/15/2017 --    JP v0.1.5
*
*/
#include "layer.h"


pooling::pooling() {}
pooling::~pooling() {
}
// windowsize is set to zero if overlap is false or windowsize is not passed to function
void pooling::init(string namelayer, int method, int steps, bool isoverlap, string resformat, int windowsize = 0) {
	layerType = "pooling";
	layerName = namelayer;
	resFormat = resformat;
	step = steps;
	windowSize = windowsize;
	poolingMethod = method;
	overlap = isoverlap;
	if (overlap == false && windowSize != 0)
	{
		cout << "Warning : isOverlap = false, windowsize must be 0" << endl;
		cout << "Continuning with overlaping set to false windowsize is set 0" << endl;
		overlap = false;
		windowSize = 0;
	}

	if (overlap == true && windowSize == 0)
	{
		cout << "Warning : isOverlap = true, windowsize cannot be 0" << endl;
		cout << "Continuning with overlaping set to false windowsize is set 0" << endl;
		overlap = false;
		windowSize = 0;
	}
}

void pooling::feedForward(int numOfSamples, layer* prevLayer, bool train ) {

	if (prevLayer->resFormat == "matrix") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		cout << "Pooling cannot be performed with matrix" << endl;
		return;
	}
	std::vector<std::vector<Mat> > input(prevLayer->resVector);
	location.clear();
	resVector.clear();
	location.resize(prevLayer->resVector.size());
	resVector.resize(prevLayer->resVector.size());
	for (int i = 0; i < prevLayer->resVector.size(); i++) {
		location[i].resize(prevLayer->resVector[i].size());
		resVector[i].resize(prevLayer->resVector[i].size());
	}
	if (overlap) {
		Size2i osize = Size(windowSize, windowSize);
		for (int i = 0; i < input.size(); i++) {
			for (int j = 0; j < input[i].size(); j++) {
				if (train) {
					Mat tmp = PoolingOverlap(input[i][j], osize, step, poolingMethod, location[i][j]);
					tmp.copyTo(resVector[i][j]);
				}
				else {
					Mat tmp = PoolingOverlapTest(input[i][j], osize, step, poolingMethod);
					tmp.copyTo(resVector[i][j]);
				}
				
			}
		}
	}
	else {
		for (int i = 0; i < input.size(); i++) {
			for (int j = 0; j < input[i].size(); j++) {
				if (train)
				{
					Mat tmp = Pooling(input[i][j], step, poolingMethod, location[i][j]);
					tmp.copyTo(resVector[i][j]);
				}
				else
				{
					Mat tmp = PoolingTest(input[i][j], step, poolingMethod);
					tmp.copyTo(resVector[i][j]);
				}
				
			}
		}
	}
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);
}

void pooling::backprop(int numOfSamples, layer* prevLayer, layer* nextLayer) {


	if (prevLayer->resFormat != "image") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		return;
	}
	std::vector<std::vector<Mat> > derivative;
	std::vector<std::vector<Mat> > secDerivative;
	std::vector<std::vector<Mat> > input(prevLayer->resVector);
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
	Size2i upSize = Size(input[0][0].rows, input[0][0].cols);
	Mat tmp;
	if (overlap) {
		Size2i osize = Size(windowSize, windowSize);
		for (int i = 0; i < derivative.size(); i++) {
			for (int j = 0; j < derivative[i].size(); j++) {
				tmp = revPoolingOverlap(derivative[i][j], osize, step, poolingMethod, location[i][j], upSize);
				tmp.copyTo(deltaVector[i][j]);
				tmp = revPoolingOverlap(secDerivative[i][j], osize, step, poolingMethod, location[i][j], upSize);
				tmp.copyTo(secDvector[i][j]);
			}
		}
	}
	else {
		for (int i = 0; i < derivative.size(); i++) {
			for (int j = 0; j < derivative[i].size(); j++) {
				tmp = revPooling(derivative[i][j], step, poolingMethod, location[i][j], upSize);
				tmp.copyTo(deltaVector[i][j]);
				tmp = revPooling(secDerivative[i][j], step, poolingMethod, location[i][j], upSize);
				tmp.copyTo(secDvector[i][j]);
			}
		}
	}
	derivative.clear();
	std::vector<std::vector<Mat> >().swap(derivative);
	secDerivative.clear();
	std::vector<std::vector<Mat> >().swap(secDerivative);
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);
}

