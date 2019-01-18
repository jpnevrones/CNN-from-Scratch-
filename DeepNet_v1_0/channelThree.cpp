#include "DeepNet_v1_0.h"
#include "channelThree.h"

using namespace std;
using namespace cv;

Mat parallel(func f, const Mat &matrix)
{
	vector<Mat> matrixCH3;
	split(matrix, matrixCH3);
	vector<Mat> vecRes;
	Mat res = Mat::zeros(matrix.rows, matrix.cols, CV_64FC3);
	for (int i = 0; i < 3; i++)
	{
		vecRes.push_back(f(matrixCH3[i]));
	}
	merge(vecRes, res);
	matrixCH3.clear();
	vecRes.clear();
	return  res;
}

Mat parallel(func2 f, const Mat &matrix1, const Mat &matrix2)
{
	vector<Mat> matrix1CH3;
	vector<Mat> matrix2CH3;
	split(matrix1, matrix1CH3);
	split(matrix2, matrix2CH3);
	vector<Mat> vecRes;
	Mat res;
	for (int i = 0; i < matrix1CH3.size(); i++)
	{
		vecRes.push_back(f(matrix1CH3[i], matrix2CH3[i]));
	}
	merge(vecRes, res);
	matrix1CH3.clear();
	matrix2CH3.clear();
	vecRes.clear();
	return res;
}

Mat parallel(func3 f, const Mat &matrix1, const Mat &matrix2, int a)
{
	vector<Mat> matrix1CH3;
	vector<Mat> matrix2CH3;
	if (matrix1.channels() == 3) split(matrix1, matrix1CH3);
	else { for (int i = 0; i < 3; i++) matrix1CH3.push_back(matrix1); }
	if (matrix2.channels() == 3) split(matrix2, matrix2CH3);
	else { for (int i = 0; i < 3; i++) matrix2CH3.push_back(matrix2); }
	vector<Mat> vecRes;
	Mat res;
	for (int i = 0; i < 3; i++)
	{
		vecRes.push_back(f(matrix1CH3[i], matrix2CH3[i], a));
	}
	merge(vecRes, res);
	matrix1CH3.clear();
	matrix2CH3.clear();
	vecRes.clear();
	return res;
}

Mat parallel(func4 f, const Mat &matrix, int a)
{
	vector<Mat> matrixCH3;
	split(matrix, matrixCH3);
	vector<Mat> vecRes;
	Mat res = Mat::zeros(matrix.rows, matrix.cols, CV_64FC3);
	for (int i = 0; i < matrixCH3.size(); i++)
	{
		vecRes.push_back(f(matrixCH3[i], a));
	}
	merge(vecRes, res);
	matrixCH3.clear();
	vecRes.clear();
	return res;
}

Mat parallel(func5 f, const Mat &matrix1, const Mat &matrix2, int a, int bias, int c)
{
	vector<Mat> matrix1CH3;
	vector<Mat> matrix2CH3;
	if (matrix1.channels() == 3) split(matrix1, matrix1CH3);
	else { for (int i = 0; i < 3; i++) matrix1CH3.push_back(matrix1); }
	if (matrix2.channels() == 3) split(matrix2, matrix2CH3);
	else { for (int i = 0; i < 3; i++) matrix2CH3.push_back(matrix2); }
	vector<Mat> vecRes;
	Mat res;
	for (int i = 0; i < 3; i++)
	{
		vecRes.push_back(f(matrix1CH3[i], matrix2CH3[i], a, bias, c));
	}
	merge(vecRes, res);
	matrix1CH3.clear();
	matrix2CH3.clear();
	vecRes.clear();
	return res;
}

Mat parallel(func6 f, const Mat &matrix, int a, int bias)
{
	vector<Mat> matrixCH3;
	split(matrix, matrixCH3);
	vector<Mat> vecRes;
	Mat res = Mat::zeros(matrix.rows, matrix.cols, CV_64FC3);
	for (int i = 0; i < matrixCH3.size(); i++)
	{
		vecRes.push_back(f(matrixCH3[i], a, bias));
	}
	merge(vecRes, res);
	matrixCH3.clear();
	vecRes.clear();
	return res;
}