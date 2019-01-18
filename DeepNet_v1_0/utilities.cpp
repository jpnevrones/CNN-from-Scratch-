/*	utilities.cpp : source file - contains the matrix based compuation utilities 
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

/*Matrix based Computation utilities
* changes added on 03/31/2017 -- JP v0.1
*/

#include "DeepNet_v1_0.h"


Mat reciprocal(const Mat &matrix)
{
	if (matrix.channels() == 1) return 1.0 / matrix;
	else {
		Mat one = Mat(matrix.size(), CV_64FC3, Scalar(1.0, 1.0, 1.0));
		return divMat(one, matrix);
	}

}
Mat sigmoid(const Mat & matrix) 
{
	Mat res = expMat(-matrix) + 1.0;
	return div(1.0, res);
}
Mat rotBy90(const Mat & matrix, int k)
{
	Mat res;
	if (k == 0) return matrix;
	else if (k == 1) {
		flip(matrix.t(), res, 0);
	}
	else {
		flip(rotBy90(matrix, k - 1).t(), res, 0);
	}
	return res;

}

/*activation function for neural network computation*/

//Relu
Mat relu(const Mat& matrix)
{
	Mat res = matrix > 0.0;
	res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
	res = res.mul(matrix);
	return res;
}
//Leaky Relu
Mat leakyRelu(const Mat& matrix)
{
	Mat res1 = matrix > 0.0;
	Mat res2 = matrix < 0.0;

	res1.convertTo(res1, CV_64FC1, 1.0 / 255.0, 0);
	res1 = res1.mul(matrix);
	
	res2.convertTo(res2, CV_64FC1, 1.0 / 255.0, 0);
	res2 = res2.mul(matrix);
	res2 = res2.mul(1 / LRALPHA);

	return (res1 + res2);
}
Mat tanh(const Mat & matrix)
{
	Mat res;
	matrix.copyTo(res);
	for (int i = 0; i<res.rows; i++) {
		for (int j = 0; j<res.cols; j++) {
			res.ATD(i, j) = tanh(matrix.ATD(i, j));
		}
	}
	return res;
}


//Derivatives
Mat dbydxSigmoid(const Mat & matrix)
{
	Mat tmp = expMat(matrix);
	Mat tmp2 = tmp + 1.0;
	tmp2 = powMat(tmp2, 2.0);
	return divMat(tmp, tmp2);
}
Mat dbydxTanh(const Mat & matrix)
{
	Mat res = Mat::ones(matrix.rows, matrix.cols, CV_64FC1);
	return res - matrix.mul(matrix);
}
Mat dbydxRelu(const Mat& matrix)
{
	Mat res = matrix > 0.0;
	res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
	return res;
}
Mat dbydxLeakyReLU(const Mat& matrix) {
	Mat res1 = matrix > 0.0;
	Mat res2 = matrix < 0.0;

	res1.convertTo(res1, CV_64FC1, 1.0 / 255.0, 0);
	
	res2.convertTo(res2, CV_64FC1, 1.0 / 255.0, 0);
	res2 = res2.mul(1 / LRALPHA);
	return (res1 + res2);
}


/* OpenCV required for these functions
*  changes added on 04/04/2017 -- JP v0.1
*/

Scalar div(const Scalar& scala, double deno)
{
	if (deno == 0.0) return scala;
	Scalar res(0.0, 0.0, 0.0);
	for (int i = 0; i < 3; i++) {
		res[i] = scala[i] / deno;
	}
	return res;
}

double sumMat(const Mat& matrix)
{
	double res = 0.0;
	Scalar tmp = sum(matrix);
	for (int i = 0; i < matrix.channels(); i++) {
		res += tmp[i];
	}
	return res;
}

Mat expMat(const Mat& matrix)
{
	Mat res;
	exp(matrix, res);
	return res;
}

Mat logMat(const Mat& matrix)
{
	Mat res;
	log(matrix, res);
	return res;
}

Mat divMat(const Mat& matrix1, const Mat& matrix2)
{
	Mat res;
	divide(matrix1, matrix2, res);
	return res;
}

Mat div(double x, const Mat &src) 
{
	Mat res;
	src.copyTo(res);
	for (int i = 0; i < res.rows; i++) 
	{
		for (int j = 0; j < res.cols; j++) 
		{
			if (src.channels() == 3) 
			{
				for (int ch = 0; ch < 3; ch++) 
				{
					if (res.AT3D(i, j)[ch] != 0.0) res.AT3D(i, j)[ch] = x / res.AT3D(i, j)[ch];
				}
			}
			else 
			{
				if (res.ATD(i, j) != 0.0) res.ATD(i, j) = x / res.ATD(i, j);
			}
		}
	}
	return res;
}
Mat div(const Mat &src, double x) 
{
	if (x == 0.0) return src;
	Mat res;
	src.copyTo(res);
	for (int i = 0; i < res.rows; i++) 
	{
		for (int j = 0; j < res.cols; j++) 
		{
			if (src.channels() == 3) 
			{
				for (int ch = 0; ch < 3; ch++) 
				{
					res.AT3D(i, j)[ch] = res.AT3D(i, j)[ch] / x;
				}
			}
			else 
			{
				res.ATD(i, j) = res.ATD(i, j) / x;
			}
		}
	}
	return res;
}

Mat reduceMat(const Mat& matrix, int direc , int conf )
{
	Mat res;
	reduce(matrix, res, direc, conf);
	return res;
}

Mat powMat(const Mat& matrix, double val)
{
	Mat res;
	pow(matrix, val, res);
	return res;
}

double max(const Mat& matrix)
{
	Point min;
	Point max;
	double minval;
	double maxval;
	minMaxLoc(matrix, &minval, &maxval, &min, &max);
	return maxval;
}

double min(const Mat& matrix)
{
	Point min;
	Point max;
	double minval;
	double maxval;
	minMaxLoc(matrix, &minval, &maxval, &min, &max);
	return minval;
}

/*More utilities for network layer */

Point findLoc(const Mat &matrix, int val)
{
	Mat temp, idx;
	Point res = Point(0, 0);
	matrix.reshape(0, 1).copyTo(temp);
	sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
	int i = idx.at<int>(0, val);
	res.x = i % matrix.cols;
	res.y = i / matrix.cols;
	return res;
}

std::vector<Point> findLocCh(const Mat &matrix, int val)
{
	std::vector<Mat> matrixV;
	split(matrix, matrixV);
	std::vector<Point> res;
	for (int i = 0; i < matrixV.size(); i++)
	{
		res.push_back(findLoc(matrixV[i], val));
	}
	matrixV.clear();
	std::vector<Mat>().swap(matrixV);
	return res;
}

Mat findMax(const Mat &matrix) 
{
	Mat tmp;
	matrix.copyTo(tmp);
	Mat result = Mat::zeros(1, tmp.cols, CV_64FC1);
	double minValue, maxValue;
	Point minLoc, maxLoc;
	for (int i = 0; i < tmp.cols; i++) 
	{
		minMaxLoc(tmp(Rect(i, 0, 1, tmp.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
		result.ATD(0, i) = (double)maxLoc.y;
	}
	return result;
}


void minMaxLoc(const Mat &img, Scalar &minVal, Scalar &maxVal, std::vector<Point> &minLoc, 
	std::vector<Point> &maxLoc) 
{
	std::vector<Mat> imgs;
	split(img, imgs);
	for (int i = 0; i < imgs.size(); i++) 
	{
		Point min;
		Point max;
		double minval;
		double maxval;
		minMaxLoc(imgs[i], &minval, &maxval, &min, &max);
		minLoc.push_back(min);
		maxLoc.push_back(max);
		minVal[i] = minval;
		maxVal[i] = maxval;
	}
}

int cmpMat(const Mat &matrix1, const Mat &matrix2) 
{
	Mat tmp;
	matrix2.copyTo(tmp);
	tmp -= matrix1;
	Mat res = (tmp == 0.0);
	res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
	return (int)(sumMat(res));
}


/*Data type conversion codes*/

// int to string
string i2str(int num) {
	stringstream ss;
	ss << num;
	string s = ss.str();
	return s;
}

// string to int
int str2i(string str) {
	return atoi(str.c_str());
}

Vec3d Scalar2Vec3d(Scalar a) {
	Vec3d res(a[0], a[1], a[2]);
	return res;
}

Scalar Vec3d2Scalar(Vec3d a) {
	return Scalar(a[0], a[1], a[2]);
}

Scalar makeScalar(double matrix) {
	Scalar res = Scalar(matrix, matrix, matrix);
	return res;
}

// convert from vector of img to matrix
// vec.size() == numOfSamples
void convert(std::vector<std::vector<Mat> >& vec, Mat &matrix) {
	int subFeatures = vec[0][0].rows * vec[0][0].cols;
	Mat res = Mat::zeros(3 * vec[0].size() * subFeatures, vec.size(), CV_64FC1);
	for (int i = 0; i < vec.size(); i++) {
		for (int matrix = 0; matrix < vec[i].size(); matrix++) {

			std::vector<Mat> tmpvec;
			split(vec[i][matrix], tmpvec);
			for (int j = 0; j < tmpvec.size(); j++) {
				Rect roi = Rect(i, matrix * 3 * subFeatures + j * subFeatures, 1, subFeatures);
				Mat subView = res(roi);
				Mat tmp = tmpvec[j].reshape(0, subFeatures);
				tmp.copyTo(subView);
			}
		}
	}
	res.copyTo(matrix);
}

// convert from matrix to vector of img
// vec.size() == numOfSamples
void convert(Mat &matrix, std::vector<std::vector<Mat> >& vec, int numOfSamples, int imagesize) {
	std::vector<Mat> tmpvec;
	for (int i = 0; i < numOfSamples; i++) {
		tmpvec.clear();
		int dim = imagesize * imagesize;
		vector<Mat> mats;
		for (int j = 0; j < matrix.rows; j += dim * 3) {
			mats.clear();
			for (int k = 0; k < 3; k++) {
				Mat tmp;
				matrix(Rect(i, j + k * dim, 1, dim)).copyTo(tmp);
				tmp = tmp.reshape(0, imagesize);
				mats.push_back(tmp);
			}
			Mat res;
			merge(mats, res);
			tmpvec.push_back(res);
		}
		vec.push_back(tmpvec);
	}
	tmpvec.clear();
	std::vector<Mat>().swap(tmpvec);
}






