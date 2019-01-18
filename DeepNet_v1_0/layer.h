#pragma once 
#include "DeepNet_v1_0.h"

using namespace std;
using namespace cv;
class input : public layer
{
public:
	input();
	~input();
	void init(string, int, string);
	void feedForward(int, const vector<Mat>&, const Mat&, bool);
	void getSample(const std::vector<Mat>&, std::vector<vector<Mat> >&, const Mat&, Mat&);
	void backprop();

	Mat label;
	int batchSize;

};

class convolution : public layer {
public:
	convolution();
	~convolution();
	void init(string, int, int, int, int, int, double, string);
	void initWeight(layer*);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);

	std::vector<convkernel*> kernels;
	Mat totWeight;
	Mat totWeightGradient;
	Mat totWeightd2;
	int padding;
	int step;
	int featureMap;

	// updater    
	void setMomentum();
	void update(int);
	double momtumdbydx;
	double momtumSecD;
	int iter;
	double mu;
	std::vector<Mat> velocityWeight;
	std::vector<Scalar> velocityBias;
	std::vector<Mat> weightSecD;
	std::vector<Scalar> biasSecD;
	Mat totVelocityWeight;
	Mat totWeightSecD;
	Mat lRateWeight;
	Scalar lRateBias;

};

class fullyConnected : public layer
{
public:
	fullyConnected();
	~fullyConnected();
	void init(string, int, double, string);// WOULD NEED TO OVERLOAD ACCORSDINGLY
	void initWeight(layer*);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);

	Mat weight;
	Mat bias;
	Mat weightGradient;
	Mat biasGradient;
	Mat weightD2;
	Mat biasD2;

	int size;
	double weightDecay;

	// updater

	void setMomentum();
	void update(int);
	double momtumdbydx;
	double momtumSecD;
	int iter;
	double mu;
	Mat velocityWeight;
	Mat velocityBias;
	Mat weightSecD;
	Mat biasSecD;
	Mat learning_rate;

};

class softmax : public layer
{
public:
	softmax();
	~softmax();
	void init(string, int, double, string);
	void initWeight(layer*);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, Mat&);

	Mat weight;
	Mat bias;
	Mat weightGradient;
	Mat biasGradient;
	Mat weightD2;
	Mat biasD2;
	double networkCost;
	int resSize;
	double weightDecay;

	// updater
	void setMomentum();
	void update(int);

	double momtumdbydx;
	double momtumSecD;
	int iter;
	double mu;
	Mat velocityWeight;
	Mat velocityBias;
	Mat weightSecD;
	Mat biasSecD;
	Mat learning_rate;

};

class activation : public layer {
public:
	activation();
	~activation();
	void init(string, int, string);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);
	int method;
};



class pooling : public layer
{
public:
	pooling();
	~pooling();

	void init(string, int, int, bool, string, int);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);

	int step;
	int windowSize;
	int poolingMethod;
	bool overlap;
	std::vector<std::vector<std::vector<std::vector<Point> > > > location;

};

class dropout : public layer
{
public:
	dropout();
	~dropout();
	void init(string, double, string);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);

	double dropoutRate;
	Mat bernoulliMatrix;
	std::vector<std::vector<Mat> > bernoulliVector;

};

class normalization : public layer {
public:

	normalization();
	~normalization();
	void init(string, double, double, double, int, string);
	void feedForward(int, layer*, bool);
	void backprop(int, layer*, layer*);
	Mat normalize(std::vector<Mat>&, int);
	Mat normalizedbybdx(std::vector<Mat>&, int);

	double alpha;
	double beta;
	double k;
	int n;
};


