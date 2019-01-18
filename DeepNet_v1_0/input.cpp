/*	input.h : Header file - input Network layer
*	version 0.1.2
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	-	Initial draft -- 04/04/2017 --    JP v0.1.1
*	-	Adding more input processing support and integration to core network function -- 04/12/2017 --    JP v0.1.2
*			
*
*/
#include "layer.h"

input::input() {}
input::~input() {}


void input::init(string namelayer, int batchsize, string resformat) 
{
	layerType = "input";
	layerName = namelayer;
	batchSize = batchsize;
	resFormat = resformat;
	label = Mat::zeros(1, batchSize, CV_64FC1);
}

void input::feedForward(int numOfSamples, const vector<Mat>& inputData, 
	const Mat& inputLabel, bool train)
{
	if (train)
	{
		getSample(inputData, resVector, inputLabel, label);
	}
	else
	{
		resVector.resize(inputData.size());
		for (int i = 0; i < resVector.size(); i++) {
			resVector[i].resize(1);
		}
		for (int i = 0; i < inputData.size(); i++) {
			inputData[i].copyTo(resVector[i][0]);
		}

		inputLabel.copyTo(label);
	}
	
}

void input::getSample(const vector<Mat>& matrix1, vector<vector<Mat> >& resmatrix1, 
	const Mat& matrix2, Mat& resmatrix2) {

	resmatrix1.clear();
	if (isGradientTrue) {
		for (int i = 0; i < batchSize; i++) {
			vector<Mat> tmp;
			tmp.push_back(matrix1[i]);
			resmatrix1.push_back(tmp);
			resmatrix2.ATD(0, i) = matrix2.ATD(0, i);
		}
		return;
	}
	if (matrix1.size() < batchSize) {
		for (int i = 0; i < matrix1.size(); i++) {
			vector<Mat> tmp;
			tmp.push_back(matrix1[i]);
			resmatrix1.push_back(tmp);
		}
		Rect roi = Rect(0, 0, matrix2.cols, 1);
		matrix2(roi).copyTo(resmatrix2);
		return;
	}
	vector<int> sample_vec;
	for (int i = 0; i < matrix1.size(); i++) {
		sample_vec.push_back(i);
	}
	random_shuffle(sample_vec.begin(), sample_vec.end());
	for (int i = 0; i < batchSize; i++) {
		vector<Mat> tmp;
		tmp.push_back(matrix1[sample_vec[i]]);
		resmatrix1.push_back(tmp);
		resmatrix2.ATD(0, i) = matrix2.ATD(0, sample_vec[i]);
	}
}

void input::backprop() {;
// won't require backward pass for input network layer
}


