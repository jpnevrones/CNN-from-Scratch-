/*	Network.h : Header file - Network and layer based support required by Neural Network framework
*	Supported Network layer type: input, convolutional, full_connected, pooling, softmax, dropout.
*   Max , stochastic pooling and poooling with overlap supported.
*	version 0.1.5
*	Developers : JP - Jithin Pradeep(jxp161430),
*	Last Update:
*	Initial draft by JP - Jithin Pradeep(jxp161430) on 03/31/2017 --    v0.1
*		- Base Layer class function definition -- 04/03/2017 -- JP v0.1.1
*		- Core network function -- 04/05/2017 -- JP v0.1.2
*		- Secondary Network function for traning and testing network -- 04/07/2017 -- JP v0.1.3
*		- Convolution function  -- 04/10/2017 -- JP v0.1.4
*		- Max and stochastic pooling function 
*		- normalization and activation function support for network layer
*		- overlaped pooling support for the network 				-- 04/15/2017 -- JP v0.1.5
*
*/

using namespace std;
using namespace cv;




/* Base Layer class
*  Supported Network layer type: input, convolutional, full_connected, pooling, softmax, dropout.
*/
class layer
{
public:
	layer();
	virtual ~layer();

	Mat resMatrix;
	Mat deltaMatrix;
	Mat secDmatrix;

	std::vector<std::vector<Mat> > resVector;
	std::vector<std::vector<Mat> > deltaVector;
	std::vector<std::vector<Mat> > secDvector;

	std::string layerName;
	std::string layerType;
	std::string resFormat;

};





/*Core Network function*/
void networkInit(const std::vector<Mat>&, const Mat&, std::vector<layer*>&);
void feedForward(const std::vector<Mat>&, const Mat&, std::vector<layer*>&);
void feedForwardTest(const std::vector<Mat> &, const Mat &, std::vector<layer*> &);
void backprop(std::vector<layer*>&);
void updateNetwork(std::vector<layer*>&, int);


void testDeepNet(const std::vector<Mat>&, const Mat&, std::vector<layer*>&);
void trainDeepNet(const std::vector<Mat>&, const Mat&, const std::vector<Mat>&,
	const Mat&, std::vector<layer*>&);

/*Convolution kernel*/
class convkernel
{
public:
	convkernel();
	~convkernel();
	void init(int, double);

	Mat weight;
	Scalar bias;
	Mat weightGradient;
	Scalar biasGradient;
	Mat weightD2;
	Scalar biasD2;
	int kernelSize;
	double weightDecay;

};


/*Convolution function*/
Mat convolution2D(const Mat&, const Mat&, int, int, int);

UMat convolutionDFT(const UMat&, const UMat&);
Mat convolutionDFT(const Mat&, const Mat&);
Mat convolution2DDFT(const Mat&, const Mat&, int, int, int);

Mat convolutionCalculas (const Mat&, const Mat&, int, int, int);
Mat Padding(Mat&, int);
Mat revPadding(Mat&, int);
Mat interpolation(Mat&, Mat&);
Mat interpolation(Mat&, int);
Mat kroneckerProduct(const Mat&, const Mat&);
Mat getBernoulliMatrix(int, int, double);

/* Pooling with overlap : Max pooling and stochastic pooling supported*/

Mat Pooling(const Mat&, int, int, std::vector<std::vector<Point> > &);
Mat PoolingTest(const Mat&, int, int);

Mat PoolingOverlap(const Mat&, Size2i, int, int, std::vector<std::vector<Point> >&);
Mat PoolingOverlapTest(const Mat&, Size2i, int, int);

Mat revPooling(const Mat&, int, int, std::vector<std::vector<Point> >&, Size2i);
Mat revPoolingOverlap(const Mat&, Size2i, int, int, std::vector<std::vector<Point> >&, 
	Size2i);

/*DeepNet Set layer function : are used to insert a layer into the network architecture*/


void inputLayer(std::vector<layer*> &, string, int);
void convolutionLayer(std::vector<layer*> &, string , int , int , int , int , int , double);
void fullyConnectedLayer(std::vector<layer*> &, string , int , double);
void poolingFun(std::vector<layer*> &, string , string , int,  bool , int);
void softmaxLayer(std::vector<layer*> &, string , int , double);
void normalize(std::vector<layer*> &, string , double , double , double , double);
void activationFun(std::vector<layer*> &, string , string);
void dropoutLayer(std::vector<layer*> &, string , double);












