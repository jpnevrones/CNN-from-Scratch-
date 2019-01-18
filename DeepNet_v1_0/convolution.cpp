/*	convolution.h : Header file - convolution Network layer
*	version 0.1.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*		Initial draft -- 04/04/2017 --    JP v0.1.1
*		- Convolution base class template -- 04/04/2017 --    JP v0.1.1
*		- Convolution kernel -- 04/10/2017 --    JP v0.1.2
*		- Convolution Feedforward -- 04/10/2017 --    JP v0.1.3
*		- Convolution update -- 04/11/2017 --    JP v0.1.4
*		- Convolution backprop logic -- 04/13/2017 --    JP v0.1.5
*		- More updates  to Convolution feedforward backprop and update -- 04/14/2017 --    JP v0.1.6
*
*/
#include "DeepNet_v1_0.h"


// Convolution kernel
convkernel::convkernel() {}
convkernel::~convkernel() {}

void convkernel::init(int width, double weightdecay) {
	kernelSize = width;
	weightDecay = weightdecay;
	weight = cv::Mat(kernelSize, kernelSize, CV_64FC3, Scalar::all(1.0));
	randu(weight, Scalar::all(-1.0), Scalar::all(1.0));
	bias = Scalar::all(0.0);
	weightGradient = Mat::zeros(weight.size(), CV_64FC3);
	biasGradient = Scalar::all(0.0);
	weightD2 = Mat::zeros(weight.size(), CV_64FC3);
	biasD2 = Scalar::all(0.0);

	double epsilon = 0.12;
	weight = weight * epsilon;
}

// layer 
convolution::convolution() {}
convolution::~convolution() {
	kernels.clear();
	vector<convkernel*>().swap(kernels);
}

void convolution::init(string namelayer, int Numkernel, int kernelSize, int fMaps, int paddings, int steps, double weightdecay, string resformat) {
	layerType = "convolutional";
	layerName = namelayer;
	resFormat = resformat;
	padding = paddings;
	step = steps;
	featureMap = fMaps;

	kernels.clear();
	for (int i = 0; i < Numkernel; ++i) {
		convkernel *tmpCovKernel = new convkernel();
		tmpCovKernel->init(kernelSize, weightdecay);
		kernels.push_back(tmpCovKernel);
	}
}

void convolution::initWeight(layer* prevLayer) {

	if (featureMap > 0) {
		totWeight = Mat::ones(kernels.size(), featureMap, CV_64FC1);
		totWeightGradient = Mat::zeros(totWeight.size(), CV_64FC1);
		totWeightd2 = Mat::zeros(totWeight.size(), CV_64FC1);
	}

	// updater
	Mat tmpWeight = Mat::zeros(kernels[0]->weight.size(), CV_64FC3);
	totVelocityWeight = Mat::zeros(totWeight.size(), CV_64FC1);
	totWeightSecD = Mat::zeros(totWeight.size(), CV_64FC1);

	velocityWeight.resize(kernels.size());
	velocityBias.resize(kernels.size());
	weightSecD.resize(kernels.size());
	biasSecD.resize(kernels.size());
	for (int i = 0; i < kernels.size(); ++i) {
		tmpWeight.copyTo(velocityWeight[i]);
		tmpWeight.copyTo(weightSecD[i]);
		velocityBias[i] = Scalar::all(0.0);
		biasSecD[i] = Scalar::all(0.0);
	}
	iter = 0;
	mu = 1e-2;
	convolution::setMomentum();
	tmpWeight.release();
}

void convolution::setMomentum() {
	if (iter < 30) {
		momtumdbydx = momtumWeightInit;
		momtumSecD = momtumSecDevInit;
	}
	else {
		momtumdbydx = momtumWeightAdj;
		momtumSecD = momtumSecDevAdj;
	}
}

void convolution::update(int numOfIteration) {
	iter = numOfIteration;
	if (iter == 30) convolution::setMomentum();

	for (int i = 0; i < kernels.size(); ++i) {
		weightSecD[i] = momtumSecD * weightSecD[i] + (1.0 - momtumSecD) * kernels[i]->weightD2;
		lRateWeight = lrateweight / (weightSecD[i] + Scalar::all(mu));
		velocityWeight[i] = velocityWeight[i] * momtumdbydx + (1.0 - momtumdbydx) * kernels[i]->weightGradient.mul(lRateWeight);
		kernels[i]->weight -= velocityWeight[i];

		biasSecD[i] = momtumSecD * biasSecD[i] + (1.0 - momtumSecD) * kernels[i]->biasD2;
		lRateBias = lratebias / (biasSecD[i] + Scalar::all(mu));
		velocityBias[i] = velocityBias[i] * momtumdbydx + (1.0 - momtumdbydx) * kernels[i]->biasGradient.mul(lRateBias);
		kernels[i]->bias -= velocityBias[i];
	}
	if (featureMap > 0) {
		totWeightSecD = momtumSecD * totWeightSecD + (1.0 - momtumSecD) * totWeightd2;
		lRateWeight = lrateweight / (totWeightSecD + mu);
		totVelocityWeight = totVelocityWeight * momtumdbydx + (1.0 - momtumdbydx) * totWeightGradient.mul(lRateWeight);
		totWeight -= totVelocityWeight;
	}
}

void convolution::feedForward(int numOfSamples, layer* prevLayer, bool train) {
//Feedforward is same for test and train pahse, bool paarmeter is added to keep all feedforward layer function to have common syntax
	std::vector<std::vector<Mat> > input;
	if (prevLayer->resFormat == "image") {
		input = prevLayer->resVector;
	}
	else {
		// no!!!!
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		return;
	}
	Mat convWeight;
	if (featureMap > 0) {
		convWeight = expMat(totWeight);
		convWeight = divMat(convWeight, repeat(reduceMat(convWeight, 0, CV_REDUCE_SUM), convWeight.rows, 1));
	}
	resVector.clear();
	for (int i = 0; i < input.size(); ++i) {
		std::vector<Mat> convInp;
		for (int j = 0; j < input[i].size(); ++j) {
			std::vector<Mat> tmpConv;
			for (int k = 0; k < kernels.size(); ++k) {
				Mat temp = rotBy90(kernels[k]->weight, 2);
				Mat tmpConv = convolutionCalculas(input[i][j], temp, CONVVALID, padding, step);
				tmpConv += kernels[k]->bias;
				tmpConv.push_back(tmpConv);
			}
			if (featureMap > 0) {
				std::vector<Mat> outputvec(featureMap);
				Mat zero = Mat::zeros(tmpConv[0].size(), CV_64FC3);
				for (int k = 0; k < outputvec.size(); k++) { zero.copyTo(outputvec[k]); }
				for (int matrix = 0; matrix < kernels.size(); matrix++) {
					for (int n = 0; n < featureMap; n++) {
						outputvec[n] += tmpConv[matrix].mul(Scalar::all(convWeight.ATD(matrix, n)));
					}
				}
				for (int k = 0; k < outputvec.size(); k++) { convInp.push_back(outputvec[k]); }
				outputvec.clear();
			}
			else {
				for (int k = 0; k < tmpConv.size(); k++) { convInp.push_back(tmpConv[k]); }
			}
			tmpConv.clear();
		}
		resVector.push_back(convInp);
	}
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);
	convWeight.release();
}

void convolution::backprop(int numOfSamples, layer* prevLayer, layer* nextLayer) {

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
	if (prevLayer->resFormat != "image") {
		cout << "Exception Caught : Invalid Result Format : resFormat invalid, expecting image not matrix" << endl;
		return;
	}
	deltaVector.clear();
	secDvector.clear();
	deltaVector.resize(prevLayer->resVector.size());
	secDvector.resize(prevLayer->resVector.size());
	for (int i = 0; i < deltaVector.size(); i++) {
		deltaVector[i].resize(prevLayer->resVector[i].size());
		secDvector[i].resize(prevLayer->resVector[i].size());
	}

	Mat tmp, tmp2, tmp3;
	std::vector<Mat> tmpWeightGradient(kernels.size());
	std::vector<Mat> tmpWeightd2(kernels.size());
	std::vector<Scalar> tmpBiasGradient;
	std::vector<Scalar> tmpBiasd2;
	tmp = Mat::zeros(kernels[0]->weight.size(), CV_64FC3);
	Scalar tmpscalar(0.0, 0.0, 0.0);
	for (int matrix = 0; matrix < kernels.size(); matrix++) {
		tmp.copyTo(tmpWeightGradient[matrix]);
		tmp.copyTo(tmpWeightd2[matrix]);
		tmpBiasGradient.push_back(tmpscalar);
		tmpBiasd2.push_back(tmpscalar);
	}

	Mat convWeight, convWeightGradient, convWeightd2;
	if (featureMap > 0) {
		convWeight = expMat(totWeight);
		convWeight = divMat(convWeight, repeat(reduceMat(convWeight, 0, CV_REDUCE_SUM), convWeight.rows, 1));
		convWeightGradient = Mat::zeros(convWeight.size(), CV_64FC1);
		convWeightd2 = Mat::zeros(convWeight.size(), CV_64FC1);
	}

	for (int i = 0; i < numOfSamples; i++) {
		for (int j = 0; j < prevLayer->resVector[i].size(); j++) {
			std::vector<Mat> sensi(kernels.size());
			std::vector<Mat> sensid2(kernels.size());
			Mat tmpDelta;
			Mat tmpD2;
			tmp = Mat::zeros(resVector[0][0].size(), CV_64FC3);
			for (int matrix = 0; matrix < kernels.size(); matrix++) {

				tmp.copyTo(sensi[matrix]);
				tmp.copyTo(sensid2[matrix]);
				if (featureMap > 0) {
					for (int n = 0; n < featureMap; n++) {
						sensi[matrix] += derivative[i][j * featureMap + n].mul(Scalar::all(convWeight.ATD(matrix, n)));
						sensid2[matrix] += secDerivative[i][j * featureMap + n].mul(Scalar::all(pow(convWeight.ATD(matrix, n), 2.0)));
					}
				}
				else {
					sensi[matrix] += derivative[i][j * kernels.size() + matrix];
					sensid2[matrix] += secDerivative[i][j * kernels.size() + matrix];
				}

				if (step > 1) {
					int len = prevLayer->resVector[0][0].rows + padding * 2 - kernels[0]->weight.rows + 1;
					sensi[matrix] = interpolation(sensi[matrix], len);
					sensid2[matrix] = interpolation(sensid2[matrix], len);
				}

				if (matrix == 0) {
					tmpDelta = convolutionCalculas(sensi[matrix], kernels[matrix]->weight, CONVFULL, 0, 1);
					tmpD2 = convolutionCalculas(sensid2[matrix], powMat(kernels[matrix]->weight, 2.0), CONVFULL, 0, 1);
				}
				else {
					tmpDelta += convolutionCalculas(sensi[matrix], kernels[matrix]->weight, CONVFULL, 0, 1);
					tmpD2 += convolutionCalculas(sensid2[matrix], powMat(kernels[matrix]->weight, 2.0), CONVFULL, 0, 1);
				}
				Mat input;
				if (padding > 0) {
					input = Padding(prevLayer->resVector[i][j], padding);
				}
				else {
					prevLayer->resVector[i][j].copyTo(input);
				}
				tmp2 = rotBy90(sensi[matrix], 2);
				tmp3 = rotBy90(sensid2[matrix], 2);
				tmpWeightGradient[matrix] += convolutionCalculas(input, tmp2, CONVVALID, 0, 1);
				tmpWeightd2[matrix] += convolutionCalculas(powMat(input, 2.0), tmp3, CONVVALID, 0, 1);
				tmpBiasGradient[matrix] += sum(tmp2);
				tmpBiasd2[matrix] += sum(tmp3);

				if (featureMap > 0) {
					// combine feature map weight matrix (after softmax)
					prevLayer->resVector[i][j].copyTo(input);
					tmp2 = rotBy90(kernels[matrix]->weight, 2);
					tmp2.copyTo(tmp3);
					tmp2 = convolutionCalculas(input, tmp2, CONVVALID, padding, step);
					tmp3 = convolutionCalculas(powMat(input, 2.0), powMat(tmp3, 2.0), CONVVALID, padding, step);
					for (int n = 0; n < featureMap; n++) {
						Mat tmpd;
						tmpd = tmp2.mul(derivative[i][j * featureMap + n]);
						convWeightGradient.ATD(matrix, n) += sumMat(tmpd);
						tmpd = tmp3.mul(secDerivative[i][j * featureMap + n]);
						convWeightd2.ATD(matrix, n) += sumMat(tmpd);
					}
				}
			}
			if (padding > 0) {
				tmpDelta = revPadding(tmpDelta, padding);
				tmpD2 = revPadding(tmpD2, padding);
			}
			tmpDelta.copyTo(deltaVector[i][j]);
			tmpD2.copyTo(secDvector[i][j]);
			sensi.clear();
			std::vector<Mat>().swap(sensi);
			sensid2.clear();
			std::vector<Mat>().swap(sensid2);
		}
	}

	for (int i = 0; i < kernels.size(); i++) {
		kernels[i]->weightGradient = div(tmpWeightGradient[i], numOfSamples) + kernels[i]->weight * kernels[i]->weightDecay;
		kernels[i]->weightD2 = div(tmpWeightd2[i], numOfSamples) + Scalar::all(kernels[i]->weightDecay);
		kernels[i]->biasGradient = div(tmpBiasGradient[i], numOfSamples);
		kernels[i]->biasD2 = div(tmpBiasd2[i], numOfSamples);
	}

	if (featureMap > 0) {
		tmp2 = convWeightGradient.mul(convWeight);
		tmp2 = repeat(reduceMat(tmp2, 0, CV_REDUCE_SUM), convWeightGradient.rows, 1);
		tmp = convWeightGradient - tmp2;
		tmp = convWeight.mul(tmp);
		tmp = div(tmp, numOfSamples);
		tmp.copyTo(totWeightGradient);

		tmp2 = convWeightd2.mul(convWeight);
		tmp2 = repeat(reduceMat(tmp2, 0, CV_REDUCE_SUM), convWeightd2.rows, 1);
		tmp = convWeightd2 - tmp2;
		tmp = convWeight.mul(tmp);
		tmp = div(tmp, numOfSamples);
		tmp.copyTo(totWeightd2);
	}
	tmpWeightGradient.clear();
	std::vector<Mat>().swap(tmpWeightGradient);
	tmpWeightd2.clear();
	std::vector<Mat>().swap(tmpWeightd2);
	derivative.clear();
	std::vector<std::vector<Mat> >().swap(derivative);
	secDerivative.clear();
	std::vector<std::vector<Mat> >().swap(secDerivative);
	tmpBiasGradient.clear();
	std::vector<Scalar>().swap(tmpBiasGradient);
	tmpBiasd2.clear();
	std::vector<Scalar>().swap(tmpBiasd2);


}


