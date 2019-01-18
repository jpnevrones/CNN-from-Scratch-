/*  normalization.h : Header file - normalization implementation 
*	version 0.1.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/14/2017 --    JP v0.1.1
*		- normalization base class template -- 04/14/2017 --    JP v0.1.2
*		- normalization Feedforward -- 04/15/2017 --    JP v0.1.3
*
*/

/*Normalization and non linearity method*/
#include "layer.h"



normalization::normalization() {}
normalization::~normalization() {
}

void normalization::init(string namelayer, double palpha, double pbeta, double pk, int pn, string resformat) {
	layerType = "normalization";
	layerName = namelayer;
	resFormat = resformat;
	alpha = palpha;
	beta = pbeta;
	k = pk;
	n = pn;
}

void normalization::feedForward(int nsamples, layer* prevLayer, bool train) {
	//Feedforward is same for test and train pahse, bool paarmeter is added to keep all feedforward layer function to have common syntax

	if (resFormat == "matrix") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		return;
	}
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
			Mat res = normalize(input[i], j);
			res.copyTo(resVector[i][j]);
		}
	}
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);
}


void normalization::backprop(int nsamples, layer* prevLayer, layer* nextLayer) {

	if (resFormat == "matrix") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		return;
	}

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
	Mat tmp, tmp2;
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
			tmp = normalizedbybdx(input[i], j);
			tmp2 = derivative[i][j].mul(tmp);
			tmp2.copyTo(deltaVector[i][j]);
			tmp2 = secDerivative[i][j].mul(powMat(tmp, 2.0));
			tmp2.copyTo(secDvector[i][j]);
		}
	}
	derivative.clear();
	std::vector<std::vector<Mat> >().swap(derivative);
	secDerivative.clear();
	std::vector<std::vector<Mat> >().swap(secDerivative);
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);

}

Mat normalization::normalize(std::vector<Mat> &vec, int which) {

	Mat res;
	vec[which].copyTo(res);
	Mat sum = Mat::zeros(res.size(), CV_64FC3);
	int from, to;
	if (vec.size() < n) {
		from = 0;
		to = vec.size() - 1;
	}
	else {
		int half = n >> 1;
		from = (k - half) >= 0 ? (k - half) : 0;
		to = (k + half) <= (vec.size() - 1) ? (k + half) : (vec.size() - 1);
	}
	for (int i = from; i <= to; ++i) {
		sum += powMat(vec[i], 2.0);
	}
	double scale = alpha / (to - from + 1);
	sum = sum.mul(Scalar::all(scale)) + Scalar::all(k);
	divide(res, powMat(sum, beta), res);
	return res;
}

Mat normalization::normalizedbybdx(std::vector<Mat> &vecInp, int which) {

	Mat input;
	vecInp[which].copyTo(input);
	Mat sum = Mat::zeros(input.size(), CV_64FC3);
	int from, to;
	if (vecInp.size() < n) {
		from = 0;
		to = vecInp.size() - 1;
	}
	else {
		int half = n >> 1;
		from = (k - half) >= 0 ? (k - half) : 0;
		to = (k + half) <= (vecInp.size() - 1) ? (k + half) : (vecInp.size() - 1);
	}
	for (int i = from; i <= to; ++i) {
		sum += powMat(vecInp[i], 2.0);
	}
	double scale = alpha / (to - from + 1);
	sum = sum.mul(Scalar::all(scale)) + Scalar::all(k);

	Mat t1 = powMat(sum, beta - 1);	// pow(sum, beta - 1)
	Mat t2 = t1.mul(sum); 			// pow(sum, beta)
	Mat t3 = t2.mul(t2); 			// pow(sum, 2*beta)

	double tmp = beta * alpha / (to - from + 1) * 2;
	Mat tmp2 = input.mul(input).mul(t1);
	tmp2 = tmp2.mul(Scalar::all(tmp));
	Mat res = t2 - tmp2;
	res = divMat(res, t3);
	return res;
}

