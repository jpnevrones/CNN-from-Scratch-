/*	utilities.h : Header file - Utility/General function required by Neural Network framework
*	version 0.1.3
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 03/31/2017 --    JP v0.1.1
*		- Matrix based Computation utilities -- 03/31/2017 -- JP v0.1.1
*		- OpenCV required for these functions -- 04/04/2017 -- JP v0.1.2
*		- More utiliries for network layer -- 04/07/2017 -- JP v0.1.3
*
*/

/*Matrix based Computation utilities*/
#pragma once
#include "DeepNet_v1_0.h"
using namespace cv;
using namespace std;

Mat reciprocal(const Mat &);
Mat sigmoid(const Mat &);
Mat relu(const Mat&);
Mat leakyRelu(const Mat&);
Mat tanh(const Mat &);
Mat rotBy90(const Mat &, int);

Mat dbydxSigmoid(const Mat &);
Mat dbydxTanh(const Mat &);
Mat dbydxRelu(const Mat&);
Mat dbydxLeakyReLU(const Mat& matrix);

/* OpenCV required for these functions*/

Scalar div(const Scalar&, double);
double sumMat(const Mat&);
Mat expMat(const Mat&);
Mat logMat(const Mat&);
Mat divMat(const Mat&, const Mat&);
Mat div(double, const Mat &);
Mat div(const Mat &, double);
Mat reduceMat(const Mat&, int, int);
Mat powMat(const Mat&, double);
double max(const Mat&);
double min(const Mat&);

/*More utilities for network layer */

Point findLoc(const Mat&, int);
std::vector<Point> findLocCh(const Mat&, int);
Mat findMax(const Mat&);
void minMaxLoc(const Mat&, Scalar&, Scalar&, std::vector<Point>&, std::vector<Point>&);
int cmpMat(const Mat&, const Mat&);



/*Data type conversion codes*/
string i2str(int);
int str2i(string);

Vec3d Scalar2Vec3d(Scalar);
Scalar Vec3d2Scalar(Vec3d);
Scalar makeScalar(double);

void convert(std::vector<std::vector<Mat> >&, Mat&);
void convert(Mat&, std::vector<std::vector<Mat> >&, int, int);



