/*	Network.cpp : source file - Network and layer based support required by Neural Network framework
*	Supported Network layer type: input, convolution, full_connected, pooling, softmax, dropout.
*	Minimal support for branching and combining layers in the network.
*   Max , stochastic pooling and poooling with overlap supported.
*	version 0.1.8
*	Developers : JP - Jithin Pradeep(jxp161430),
*	Last Update:
*	Initial draft by JP - Jithin Pradeep(jxp161430) on 03/31/2017 --    v0.1
*		- Base Layer class function definition -- 04/03/2017 -- JP v0.1.1
*		- Core network function -- 04/05/2017 -- JP v0.1.2
*		- Secondary Network function for traning and testing network -- 04/07/2017 -- JP v0.1.3
*		- Convolution function  -- 04/10/2017 -- JP v0.1.4
*		- Max and stochastic pooling function -- 04/12/2017 -- JP v0.1.5
*		- normalization and activation function support for network layer -- 04/14/2017 -- JP v0.1.6
*		- overlaped pooling support for the network -- 04/15/2017 -- JP v0.1.7
*
*/

#include "DeepNet_v1_0.h"
#include "channelThree.h"

layer::layer() {}
/*Destruct Network*/

layer::~layer()
{
	resMatrix.release();
	resVector.clear();
	std::vector<std::vector<Mat> >().swap(resVector);
}

/*Layer interface function : To insert a layer and required componeent to netwok architecture*/

void inputLayer(std::vector<layer*> &networkArch, string custLayerName, int batchSize)
{
	input *tmpLayer = new input();
	tmpLayer->init(custLayerName, batchSize, "image");

	networkArch.push_back(tmpLayer);
}

void convolutionLayer(std::vector<layer*> &networkArch, string custLayerName, int Numkernel, int convkernelSize, int featureMap,
	int paddings, int convSteps, double weightDecay)
{
	convolution *tmpLayer = new convolution();
	tmpLayer->init(custLayerName, Numkernel, convkernelSize, featureMap, paddings, convSteps, weightDecay, "image");
	networkArch.push_back(tmpLayer);
}

void fullyConnectedLayer(std::vector<layer*> &networkArch, string custLayerName, int numOfHiddenUnits, double weightDecay)
{
	fullyConnected *tmpLayer = new fullyConnected();
	tmpLayer->init(custLayerName, numOfHiddenUnits, weightDecay, "matrix");
	networkArch.push_back(tmpLayer);

}

void poolingFun(std::vector<layer*> &networkArch, string custLayerName, string poolingMethod, int poolSteps, bool isoverlap, int windowsize)
{
	pooling *tmpLayer = new pooling();
	int method;
	if (poolingMethod == "MAXPOOL") method == 0;
	else if (poolingMethod == "MEANPOOL") method == 1;
	else if (poolingMethod == "STOCHASTICPOOL") method == 2;
	else
	{
		cout << "Exception Caught : Pooling Function " << poolingMethod << " is not defined or supported " << endl;
		cout << "List of Supported pooling methods : MAXPOOL, MEANPOOL , STOCHASTICPOOL" << endl;
		cout << "Check for the spelling, all parameter are case sensitive, ensure them to be in uppercase";
		cout << "Aborting the DeepNet network intiallization, You continue by changing the pooling method " << endl;
		cout << "Press 0 - MAXPOOL" << endl << "1 - MEANPOOL" << endl << "2 - STOCHASTICPOOL " << endl;
		cin >> method;
	}

	if (method <= 2 || method >= 0)
	{
		tmpLayer->init(custLayerName, method, poolSteps, isoverlap, "image", windowsize);
		networkArch.push_back(tmpLayer);
	}
	else {
		exit(0);
		cout << "Aborting the DeepNet network intiallization, Invalid Pooling method " << endl;
	}



}

void softmaxLayer(std::vector<layer*> &networkArch, string custLayerName, int numOfOutputClass, double weightDecay)
{
	softmax *tmpLayer = new softmax();
	tmpLayer->init(custLayerName, numOfOutputClass, weightDecay, "matrix");
	networkArch.push_back(tmpLayer);

}

void normalize(std::vector<layer*> &networkArch, string custLayerName, double alpha, double beta, double k, double n)
{
	normalization *tmpLayer = new normalization();
	tmpLayer->init(custLayerName, alpha, beta, k, n, "image");
	networkArch.push_back(tmpLayer);

}

void activationFun(std::vector<layer*> &networkArch, string custLayerName, string ActivationMethod)
{
	activation *tmpLayer = new activation();
	int method;
	if (ActivationMethod == "SIGMOID") method == 0;
	else if (ActivationMethod == "TANH") method == 1;
	else if (ActivationMethod == "RELU") method == 2;
	else if (ActivationMethod == "LRELU") method == 3;
	else
	{
		cout << "Exception Caught : Activation Function " << ActivationMethod << " is not defined or supported " << endl;
		cout << "List of Supported activation : SIGMOID, TANH, RELU, LRELU" << endl;
		cout << "Check for the spelling, all parameter are case sensitive, ensure them to be in uppercase";
		cout << "Aborting the DeepNet network intiallization, You continue by changing the Activation method " << endl;
		cout << "Press 0 - SIGMOID" << endl << "1 - TANH" << endl << "2 - RELU " << endl << " 3 - LRELU " << endl;
		cin >> method;
	}

	if (method <= 3 || method >= 0)
	{
		tmpLayer->init(custLayerName, method, "matrix");
		networkArch.push_back(tmpLayer);
	}
	else {
		exit(0);
		cout << "Aborting the DeepNet network intiallization, Invalid Activation method " << endl;
	}


}

void dropoutLayer(std::vector<layer*> &networkArch, string custLayerName, double dropuoutRate)
{
	input *tmpLayer = new input();
	tmpLayer->init(custLayerName, dropuoutRate, "image");
	networkArch.push_back(tmpLayer);

}

/*Core Network function definition*/

void networkInit(const std::vector<Mat> &x, const Mat &y, std::vector<layer*> &networkArch){

	int batchSize = 0;
	cout << "Intializing DeepNet ..." << endl;
	cout << "Printing Network information " << endl;
	for (int i = 0; i < networkArch.size(); i++)	
	{
		cout << " Layer "<< i << " : " << networkArch[i]->layerType << endl;

		// input network layer
		if (networkArch[i]->layerType == "input")
		{
			batchSize = ((input*)networkArch[i])->batchSize;
			((input*)networkArch[i])->feedForward(batchSize, x, y,true);
			cout << "batch size = " << ((input*)networkArch[i])->batchSize << endl;
		}
		// convolution network layer
		else if (networkArch[i]->layerType == "convolution")	
		{
			((convolution*)networkArch[i])->initWeight(networkArch[i - 1]);
			((convolution*)networkArch[i])->feedForward(batchSize, networkArch[i - 1],true);

			cout << "kernel amount = " << ((convolution*)networkArch[i])->kernels.size() << endl;
			cout << "kernel size = " << ((convolution*)networkArch[i])->kernels[0]->weight.size() << endl;
			cout << "padding = " << ((convolution*)networkArch[i])->padding << endl;
			cout << "step = " << ((convolution*)networkArch[i])->step << endl;
			cout << "Feature map = " << ((convolution*)networkArch[i])->featureMap << endl;
			cout << "weight decay = " << ((convolution*)networkArch[i])->kernels[0]->weightDecay << endl;
		}

		// fullyConnected network layer
		else if (networkArch[i]->layerType == "fullyConnected")
		{
			((fullyConnected*)networkArch[i])->initWeight(networkArch[i - 1]);
			((fullyConnected*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);

			cout << "Hidden Size = " << ((fullyConnected*)networkArch[i])->size << endl;
			cout << "Weight Decay = " << ((fullyConnected*)networkArch[i])->weightDecay << endl;
		}

		// softmax network layer
		else if (networkArch[i]->layerType == "softmax")
		{
			((softmax*)networkArch[i])->initWeight(networkArch[i - 1]);
			((softmax*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);

			cout << "Output Size = " << ((softmax*)networkArch[i])->resSize << endl;
			cout << "Weight Decay = " << ((softmax*)networkArch[i])->weightDecay << endl;
		}

		// pooling network layer
		else if (networkArch[i]->layerType == "pooling")
		{
			((pooling*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);

			cout << "Pooling Method = " << ((pooling*)networkArch[i])->poolingMethod << endl;
			cout << "Overlap = " << ((pooling*)networkArch[i])->overlap << endl;
			cout << "Step = " << ((pooling*)networkArch[i])->step << endl;
			cout << "Window size = " << ((pooling*)networkArch[i])->windowSize << endl;
		}

		// dropout network layer
		else if (networkArch[i]->layerType == "dropout")
		{
			((dropout*)networkArch[i])->feedForward(batchSize, networkArch[i - 1],true);

			cout << "Dropout Rate = " << ((dropout*)networkArch[i])->dropoutRate << endl;
		}


		/* normalization and activation function support for network layer, changes for v0.1.5*/
		else if (networkArch[i]->layerType == "activation")
		{
			((activation*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);

			cout << "Non Linear Activation method = " << ((activation*)networkArch[i])->method << endl;
		}

		else if (networkArch[i]->layerType == "normalization")
		{
			((normalization*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
			cout << "Normalzation Parameter for Layer " << i-1 << " : " << networkArch[i-1]->layerType << endl;
			cout << "Alpha = " << ((normalization*)networkArch[i])->alpha << endl;
			cout << "Beta = " << ((normalization*)networkArch[i])->beta << endl;
			cout << "K = " << ((normalization*)networkArch[i])->k << endl;
			cout << "N = " << ((normalization*)networkArch[i])->n << endl;
		}

		if (networkArch[i]->resFormat == "matrix") {
			cout << "Resultant matrix size is " << networkArch[i]->resMatrix.size() << endl;
		}
		else {
			cout << "Resultant vector size is " << networkArch[i]->resVector.size() << " * " << networkArch[i]->resVector[0].size() << " * " << networkArch[i]->resVector[0][0].size() << endl;
		}

	}
}

void feedForward(const std::vector<Mat> &x, const Mat &y, std::vector<layer*> &networkArch)
{
	int batchSize = 0;
	double J1 = 0, J2 = 0, J3 = 0, J4 = 0;
	for (int i = 0; i < networkArch.size(); i++)
	{
		if (networkArch[i]->layerType == "input")
		{
			batchSize = ((input*)networkArch[i])->batchSize;
			((input*)networkArch[i])->feedForward(batchSize, x, y,true);
		}

		else if (networkArch[i]->layerType == "convolution")
		{
			((convolution*)networkArch[i])->feedForward(batchSize, networkArch[i - 1],true);
			// compute cost
			for (int k = 0; k < ((convolution*)networkArch[i])->kernels.size(); ++k)
			{
				J4 += sumMat(powMat(((convolution*)networkArch[i])->kernels[k]->weight, 2.0))
					* ((convolution*)networkArch[i])->kernels[k]->weightDecay / 2.0;
			}
		}

		else if (networkArch[i]->layerType == "fullyConnected")
		{
			((fullyConnected*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
			// compute cost
			J3 += sumMat(powMat(((fullyConnected*)networkArch[i])->weight, 2.0))
				* ((fullyConnected*)networkArch[i])->weightDecay / 2.0;
		}

		else if (networkArch[i]->layerType == "softmax")
		{
			((softmax*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
			// compute cost
			Mat groundTruth = Mat::zeros(((softmax*)networkArch[i])->resSize, batchSize, CV_64FC1);
			for (int i = 0; i < batchSize; i++)
			{
				groundTruth.ATD(((input*)networkArch[0])->label.ATD(0, i), i) = 1.0;
			}
			J1 += -sumMat(groundTruth.mul(logMat(networkArch[i]->resMatrix))) / batchSize;
			J2 += sumMat(powMat(((softmax*)networkArch[i])->weight, 2.0)) * ((softmax*)networkArch[i])->weightDecay / 2;
		}

		else if (networkArch[i]->layerType == "pooling")
		{
			((pooling*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
		}

		else if (networkArch[i]->layerType == "dropout")
		{
			((dropout*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
		}


		/* normalization and activation function support for network layer, changes for v0.1.5*/
		else if (networkArch[i]->layerType == "activation")
		{
			((activation*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
		}

		else if (networkArch[i]->layerType == "normalization")
		{
			((normalization*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], true);
		}

	}
	((softmax*)networkArch[networkArch.size() - 1])->networkCost = J1 + J2 + J3 + J4;
	if (!isGradientTrue)
	{
		cout << ", J1 = " << J1 << ", J2 = " << J2 << ", J3 = " << J3 << ", J4 = " << J4 << ", Cost = " << ((softmax*)networkArch[networkArch.size() - 1])->networkCost << endl;
	}
}

void feedForwardTest(const std::vector<Mat> &x, const Mat &y, std::vector<layer*> &networkArch)
{
	int batchSize = x.size();
	for (int i = 0; i < networkArch.size(); i++)
	{
		if (networkArch[i]->layerType == "input")
		{
			((input*)networkArch[i])->feedForward(batchSize, x, y, false);
		}
		else if (networkArch[i]->layerType == "convolution")
		{
			((convolution*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}
		else if (networkArch[i]->layerType == "fullyConnected")
		{
			((fullyConnected*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}
		else if (networkArch[i]->layerType == "softmax")
		{
			((softmax*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}
		else if (networkArch[i]->layerType == "pooling")
		{
			((pooling*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);

		}
		else if (networkArch[i]->layerType == "dropout")
		{
			((dropout*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}

		/* normalization and activation function support for network layer, changes for v0.1.5*/
		else if (networkArch[i]->layerType == "activation")
		{
			((activation*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}
		else if (networkArch[i]->layerType == "normalization")
		{
			((normalization*)networkArch[i])->feedForward(batchSize, networkArch[i - 1], false);
		}

	}
}

void backprop(std::vector<layer*> &networkArch) 
{

	int batchSize = ((input*)networkArch[0])->batchSize;
	Mat groundTruth = Mat::zeros(((softmax*)networkArch[networkArch.size() - 1])->resSize, batchSize, CV_64FC1);
	for (int i = 0; i < batchSize; i++) 
	{
		groundTruth.ATD(((input*)networkArch[0])->label.ATD(0, i), i) = 1.0;
	}
	for (int i = networkArch.size() - 1; i >= 0; --i) 
	{
		//cout<<networkArch[i] -> layerName<<endl;
		if (networkArch[i]->layerType == "input") 
		{
			((input*)networkArch[i])->backprop();
		}
		else if (networkArch[i]->layerType == "convolution") 
		{
			((convolution*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}
		else if (networkArch[i]->layerType == "fullyConnected") 
		{
			((fullyConnected*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}
		else if (networkArch[i]->layerType == "softmax") 
		{
			((softmax*)networkArch[i])->backprop(batchSize, networkArch[i - 1], groundTruth);
		}

		else if (networkArch[i]->layerType == "pooling") 
		{
			((pooling*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}
		else if (networkArch[i]->layerType == "dropout") 
		{
			((dropout*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}
		else if (networkArch[i]->layerType == "activation") 
		{
			((activation*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}
		else if (networkArch[i]->layerType == "normalization") 
		{
			((normalization*)networkArch[i])->backprop(batchSize, networkArch[i - 1], networkArch[i + 1]);
		}

		if (networkArch[i]->layerName == "input") break;;
		if (networkArch[i]->resFormat == "matrix") {
			//cout<<"delta dimension is "<<networkArch[i] -> deltaMatrix.size()<<endl;
		}
		else {
			//cout<<"delta dimension is "<<networkArch[i] -> deltaVector.size()<<" * "<<networkArch[i] -> deltaVector[0].size()<<" * "<<networkArch[i] -> deltaVector[0][0].size()<<endl;
		}
	}
}

void updateNetwork(std::vector<layer*> &networkArch, int iter) 
{
	for (int i = 0; i < networkArch.size(); ++i) 
	{
		//cout<<networkArch[i] -> layerName<<endl;
		if (networkArch[i]->layerType == "convolution") 
		{
			((convolution*)networkArch[i])->update(iter);
		}
		else if (networkArch[i]->layerType == "fullyConnected") 
		{
			((fullyConnected*)networkArch[i])->update(iter);
		}
		else if (networkArch[i]->layerType == "softmax") 
		{
			((softmax*)networkArch[i])->update(iter);
		}
	}
}



void testDeepNet(const std::vector<Mat> &x, const Mat &y, std::vector<layer*> &networkArch)
{

	int batchSize = 100;

	int batch_amount = x.size() / batchSize;
	int correct = 0;
	for (int i = 0; i < batch_amount; i++) 
	{
		std::vector<Mat> batchX(batchSize);
		Mat batchY = Mat::zeros(1, batchSize, CV_64FC1);
		for (int j = 0; j < batchSize; j++) 
		{
			x[i * batchSize + j].copyTo(batchX[j]);
		}
		y(Rect(i * batchSize, 0, batchSize, 1)).copyTo(batchY);
		feedForwardTest(batchX, batchY, networkArch);
		Mat res = findMax(networkArch[networkArch.size() - 1]->resMatrix);

		correct += cmpMat(res, batchY);
		batchX.clear();
		std::vector<Mat>().swap(batchX);
	}
	if (x.size() % batchSize) 
	{
		std::vector<Mat> batchX(x.size() % batchSize);
		Mat batchY = Mat::zeros(1, x.size() % batchSize, CV_64FC1);
		for (int j = 0; j < batchX.size(); j++) 
		{
			x[batch_amount * batchSize + j].copyTo(batchX[j]);
		}
		y(Rect(batch_amount * batchSize, 0, batchX.size(), 1)).copyTo(batchY);
		feedForwardTest(batchX, batchY, networkArch);
		Mat res = findMax(networkArch[networkArch.size() - 1]->resMatrix);
		correct += cmpMat(res, batchY);
		batchX.clear();
		std::vector<Mat>().swap(batchX);
	}
	cout << "correct: " << correct << ", total: " << x.size() << ", accuracy: " << double(correct) / (double)(x.size()) << endl;
}

void trainDeepNet(const std::vector<Mat> &x, const Mat &y, const std::vector<Mat> &tx, 
	const Mat &ty, std::vector<layer*> &networkArch) 
{

	networkInit(x, y, networkArch);
	

	cout << "Commencing  Traning phase  " << endl;
	int k = 0;
	for (int epo = 1; epo <= trainingEpochs; epo++) 
	{

		for (; k <= iterationPerEpoch * epo; k++) 
		{
			cout << "Epoch: " << epo << ", Iteration: " << k;//<<endl;     
			feedForward(x, y, networkArch);
			backprop(networkArch);
			updateNetwork(networkArch, k);
			//++tmpdebug;
		}
		cout << "Computing model fit using train data or in sample data: " << endl;
		testDeepNet(x, y, networkArch);
		cout << "Computing model fit using test data or out sample data: " << endl;
		testDeepNet(tx, ty, networkArch);
	}
}


/*Convolution function*/

Mat convolution2D(const Mat &img, const Mat &kernel, int convType, int padding, int step) {
	Mat tmp;
	Mat source = img;
	// padding
	source = Mat();
	copyMakeBorder(img, source, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));

	// zero padding for CONVFULL
	int additionalRows, additionalCols;
	if (CONVFULL == convType) 
	{
		additionalRows = kernel.rows - 1;
		additionalCols = kernel.cols - 1;
		copyMakeBorder(source, source, (additionalRows + 1) / 2, additionalRows / 2, 
			(additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;
	Mat fkernel;
	flip(kernel, fkernel, -1);
	filter2D(source, tmp, img.depth(), fkernel, anchor, 0, borderMode);
	// cut matrix for CONVVALID
	if (CONVVALID == convType) 
	{
		tmp = tmp.colRange((kernel.cols - 1) / 2, tmp.cols - kernel.cols / 2)
			.rowRange((kernel.rows - 1) / 2, tmp.rows - kernel.rows / 2);
	}
	int xsize = tmp.cols / step;
	if (tmp.cols % step > 0) ++xsize;
	int ysize = tmp.rows / step;
	if (tmp.rows % step > 0) ++ysize;
	Mat dest = Mat::zeros(ysize, xsize, CV_64FC1);
	for (int i = 0; i < dest.rows; i++) 
	{
		for (int j = 0; j < dest.cols; j++) 
		{
			dest.ATD(i, j) = tmp.ATD(i * step, j * step);
		}
	}
	return dest;
}

Mat convolutionDFT(const Mat &A, const Mat &bias) 
{
	Mat C;
	// reallocate the output array if needed
	C.create(abs(A.rows + bias.rows) - 1, abs(A.cols + bias.cols) - 1, A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(A.cols + bias.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + bias.rows - 1);
	// allocate temporary buffers and initialize them with 0's
	Mat tempA(dftSize, A.type(), Scalar::all(0));
	Mat tempB(dftSize, bias.type(), Scalar::all(0));
	// copy A and bias to the top-left corners of tempA and tempB, respectively
	Mat roiA(tempA, Rect(0, 0, A.cols, A.rows));
	A.copyTo(roiA);
	Mat roiB(tempB, Rect(0, 0, bias.cols, bias.rows));
	bias.copyTo(roiB);
	// now transform the padded A & bias in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, 0, A.rows);
	dft(tempB, tempB, 0, bias.rows);
	// multiply the spectrums;
	// the function handles packed spectrum representations well
	mulSpectrums(tempA, tempB, tempA, 0, false);
	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
	//idft(tempA, tempA, DFT_SCALE, A.rows + bias.rows - 1);
	// now copy the result back to C.
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
	return C;
	// all the temporary buffers will be deallocated automatically
}

UMat convolutionDFT(const UMat &A, const UMat &bias) 
{
	UMat C;
	// reallocate the output array if needed
	C.create(abs(A.rows + bias.rows) - 1, abs(A.cols + bias.cols) - 1, A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(A.cols + bias.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + bias.rows - 1);
	// allocate temporary buffers and initialize them with 0's
	UMat tempA(dftSize, A.type(), Scalar::all(0));
	UMat tempB(dftSize, bias.type(), Scalar::all(0));
	// copy A and bias to the top-left corners of tempA and tempB, respectively
	UMat roiA(tempA, Rect(0, 0, A.cols, A.rows));
	A.copyTo(roiA);
	UMat roiB(tempB, Rect(0, 0, bias.cols, bias.rows));
	bias.copyTo(roiB);
	// now transform the padded A & bias in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, 0, A.rows);
	dft(tempB, tempB, 0, bias.rows);
	// multiply the spectrums;
	// the function handles packed spectrum representations well
	mulSpectrums(tempA, tempB, tempA, 0, false);
	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
	//idft(tempA, tempA, DFT_SCALE, A.rows + bias.rows - 1);
	// now copy the result back to C.
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
	return C;
	// all the temporary buffers will be deallocated automatically
}

Mat convolution2DDFT(const Mat &img, const Mat &kernel, int convType, int padding, int step) 
{
	Mat tmp;
	Mat source;

	img.copyTo(tmp);
	// padding
	source = Mat();
	copyMakeBorder(tmp, source, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));
	/*
	UMat usource, ukernel, uconv;
	source.copyTo(usource);
	kernel.copyTo(ukernel);
	uconv = convolveDFT(usource, ukernel);
	uconv.copyTo(tmp);
	*/
	tmp = convolutionDFT(source, kernel);


	if (CONVSAME == convType) 
	{
		tmp = tmp.colRange((kernel.cols) / 2, tmp.cols - kernel.cols / 2)
			.rowRange((kernel.rows) / 2, tmp.rows - kernel.rows / 2);
	}
	if (CONVVALID == convType) 
	{
		int tmpx = source.cols - kernel.cols + 1;
		int tmpy = source.rows - kernel.rows + 1;
		tmp = tmp.colRange((tmp.cols - tmpx) / 2, tmp.cols - ((tmp.cols - tmpx) / 2))
			.rowRange((tmp.rows - tmpy) / 2, tmp.rows - ((tmp.cols - tmpx) / 2));
	}
	int xsize = tmp.cols / step;
	if (tmp.cols % step > 0) ++xsize;
	int ysize = tmp.rows / step;
	if (tmp.rows % step > 0) ++ysize;
	Mat dest = Mat::zeros(ysize, xsize, CV_64FC1);
	for (int i = 0; i < dest.rows; i++) 
	{
		for (int j = 0; j < dest.cols; j++) 
		{
			dest.ATD(i, j) = tmp.ATD(i * step, j * step);
		}
	}
	return dest;
}

Mat convolutionCalculas(const Mat &img, const Mat &kernel, int convType, int padding, int step) 
{
	Mat tmp;
	img.copyTo(tmp);
	if (tmp.channels() == 1 && kernel.channels() == 1) 
	{
		return convolution2DDFT(tmp, kernel, convType, padding, step);
	}
	else 
	{
		return parallel(convolution2DDFT, tmp, kernel, convType, padding, step);
	}
}

Mat Padding(Mat &src, int pad)
{
	Mat res;
	copyMakeBorder(src, res, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0));
	return res;
}

Mat revPadding(Mat &src, int pad) 
{
	Mat res;
	src(Rect(pad, pad, src.cols - pad * 2, src.rows - pad * 2)).copyTo(res);
	return res;
}

Mat interpolation(Mat &src, Mat &sizemat) 
{

	int step = sizemat.rows / src.rows;
	if (sizemat.rows % src.rows > 0) ++step;
	if (step == 0 || step == 1) return src;
	Mat res = Mat::zeros(sizemat.size(), CV_64FC3);
	for (int i = 0; i < src.rows; i++) 
	{
		for (int j = 0; j < src.cols; j++) 
		{
			res.AT3D(i * step, j * step) = src.AT3D(i, j);
		}
	}
	return res;
}

Mat interpolation(Mat &src, int osize) 
{
	int step = osize / src.rows;
	if (osize % src.rows > 0) ++step;
	//cout<<src.rows<<", "<<osize<<", "<<step<<endl;
	if (step == 0 || step == 1) return src;
	Mat res = Mat::zeros(osize, osize, CV_64FC3);
	for (int i = 0; i < src.rows; i++) 
	{
		for (int j = 0; j < src.cols; j++)
		{
			res.AT3D(i * step, j * step) = src.AT3D(i, j);
		}
	}
	return res;
}

// Kronecker Product 
Mat kroneckerProduct(const Mat &a, const Mat &bias) 
{
	Mat res = Mat::zeros(a.rows * bias.rows, a.cols * bias.cols, CV_64FC3);
	Mat c;
	vector<Mat> bs;
	vector<Mat> cs;
	for (int i = 0; i < a.rows; i++) 
	{
		for (int j = 0; j < a.cols; j++) 
		{
			bs.clear();
			cs.clear();
			Rect roi = Rect(j * bias.cols, i * bias.rows, bias.cols, bias.rows);
			Mat temp = res(roi);
			split(bias, bs);
			for (int ch = 0; ch < 3; ch++) 
			{
				cs.push_back(bs[ch].mul(a.AT3D(i, j)[ch]));
			}
			merge(cs, c);
			c.copyTo(temp);
		}
	}
	bs.clear();
	vector<Mat>().swap(bs);
	cs.clear();
	vector<Mat>().swap(cs);
	return res;
}

Mat getBernoulliMatrix(int height, int width, double prob) 
{
	// randu builds a Uniformly distributed matrix
	Mat ran = Mat::zeros(height, width, CV_64FC1);
	randu(ran, Scalar(0), Scalar(1.0));
	Mat res = ran >= prob;
	res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
	return res;
}


/* Pooling with overlap : Max pooling and stochastic pooling supported*/

Mat Pooling(const Mat &matrix, int step, int poolingMethod,
	std::vector<std::vector<Point> > &locat) 
{
	if (step == 1) 
	{
		std::vector<Point> tppt;
		for (int i = 0; i < matrix.rows; i++) 
		{
			for (int j = 0; j < matrix.cols; j++) 
			{
				tppt.clear();
				for (int ch = 0; ch < 3; ch++) 
				{
					tppt.push_back(Point(j, i));
				}
				locat.push_back(tppt);
			}
		}
		Mat res;
		matrix.copyTo(res);
		return res;
	}
	Mat newM;
	matrix.copyTo(newM);
	Mat res = Mat::zeros(newM.rows / step, newM.cols / step, CV_64FC3);
	for (int i = 0; i<res.rows; i++) 
	{
		for (int j = 0; j<res.cols; j++) 
		{
			Mat temp;
			Rect roi = Rect(j * step, i * step, step, step);
			newM(roi).copyTo(temp);
			Scalar val = Scalar(0.0, 0.0, 0.0);
			std::vector<Point> tppt;
			// for Max Pooling
			if (POOLMAX == poolingMethod) 
			{
				Scalar minVal = Scalar(0.0, 0.0, 0.0);
				Scalar maxVal = Scalar(0.0, 0.0, 0.0);
				std::vector<Point> minLoc;
				std::vector<Point> maxLoc;
				minMaxLoc(temp, minVal, maxVal, minLoc, maxLoc);
				val = maxVal;
				for (int ch = 0; ch < 3; ch++) 
				{
					tppt.push_back(Point(maxLoc[ch].x + j * step, maxLoc[ch].y + i * step));
				}
			}
			else if(POOLMEAN == poolingMethod) 
			{
				// Mean Pooling
				double recip = 1.0 / (step * step);
				val = sum(temp).mul(Scalar(recip, recip, recip));
				for (int ch = 0; ch < 3; ch++) {
					tppt.push_back(Point(j * step, i * step));
				}
			}
			else if(POOLSTOCHASTIC == poolingMethod) 
			{
				// Stochastic Pooling
				Scalar recip_sumval = sum(temp);
				divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
				Mat prob = temp.mul(recip_sumval);
				int ran = rand() % (temp.rows * temp.cols);
				std::vector<Point> loc = findLocCh(prob, ran);
				for (int ch = 0; ch < loc.size(); ch++) {
					val[ch] = temp.AT3D(loc[ch].y, loc[ch].x)[ch];
					tppt.push_back(Point(loc[ch].x + j * step, loc[ch].y + i * step));
				}
			}
			res.AT3D(i, j) = Scalar2Vec3d(val);
			locat.push_back(tppt);
		}
	}
	return res;
}

Mat PoolingTest(const Mat &matrix, int step, int poolingMethod) 
{
	if (step == 1) 
	{
		Mat res;
		matrix.copyTo(res);
		return res;
	}
	Mat newM;
	matrix.copyTo(newM);
	Mat res = Mat::zeros(newM.rows / step, newM.cols / step, CV_64FC3);
	for (int i = 0; i<res.rows; i++) {
		for (int j = 0; j<res.cols; j++) 
		{
			Mat temp;
			Rect roi = Rect(j * step, i * step, step, step);
			newM(roi).copyTo(temp);
			Scalar val = Scalar(0.0, 0.0, 0.0);
			// for Max Pooling
			if (POOLMAX == poolingMethod) 
			{
				Scalar minVal = Scalar(0.0, 0.0, 0.0);
				Scalar maxVal = Scalar(0.0, 0.0, 0.0);
				std::vector<Point> minLoc;
				std::vector<Point> maxLoc;
				minMaxLoc(temp, minVal, maxVal, minLoc, maxLoc);
				val = maxVal;
			}else if(POOLMEAN == poolingMethod) 
			{
				// Mean Pooling
				double recip = 1.0 / (step * step);
				val = sum(temp).mul(Scalar(recip, recip, recip));
			}else if(POOLSTOCHASTIC == poolingMethod) 
			{
				// Stochastic Pooling
				Scalar recip_sumval = sum(temp);
				divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
				Mat prob = temp.mul(recip_sumval);
				int ran = rand() % (temp.rows * temp.cols);
				std::vector<Point> loc = findLocCh(prob, ran);
				for (int ch = 0; ch < loc.size(); ch++) {
					val[ch] = temp.AT3D(loc[ch].y, loc[ch].x)[ch];
				}
			}
			res.AT3D(i, j) = Scalar2Vec3d(val);
		}
	}
	return res;
}

Mat PoolingOverlap(const Mat &matrix, Size2i windowSize, int step,
	int poolingMethod,  std::vector<std::vector<Point> > &locat) 
{
	Mat tmpres = Mat::zeros(matrix.rows - windowSize.height + 1, matrix.cols - 
		windowSize.width + 1, CV_64FC3);
	std::vector<std::vector<Point> > tmplocat;
	for (int i = 0; i < matrix.rows - windowSize.height + 1; ++i) 
	{
		for (int j = 0; j < matrix.cols - windowSize.width + 1; ++j) 
		{
			Mat tmp;
			matrix(Rect(j, i, windowSize.width, windowSize.height)).copyTo(tmp);

			Scalar val = Scalar(0.0, 0.0, 0.0);
			std::vector<Point> tppt;
			if (POOLMAX == poolingMethod) 
			{
				Scalar minVal = Scalar(0.0, 0.0, 0.0);
				Scalar maxVal = Scalar(0.0, 0.0, 0.0);
				std::vector<Point> minLoc;
				std::vector<Point> maxLoc;
				minMaxLoc(tmp, minVal, maxVal, minLoc, maxLoc);
				val = maxVal;
				for (int ch = 0; ch < 3; ch++) {
					tppt.push_back(Point(maxLoc[ch].x + j, maxLoc[ch].y + i));
				}
			}
			else if(POOLSTOCHASTIC == poolingMethod) 
			{

				Scalar recip_sumval = sum(tmp);
				divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
				Mat prob = tmp.mul(recip_sumval);
				int ran = rand() % (tmp.rows * tmp.cols);
				std::vector<Point> loc = findLocCh(prob, ran);
				for (int ch = 0; ch < loc.size(); ch++) {
					val[ch] = tmp.AT3D(loc[ch].y, loc[ch].x)[ch];
					tppt.push_back(Point(loc[ch].x + j, loc[ch].y + i));
				}
			}
			tmplocat.push_back(tppt);
			tmpres.AT3D(i, j) = Scalar2Vec3d(val);
			tppt.clear();
			std::vector<Point>().swap(tppt);
		}
	}
	int xsize = tmpres.cols / step;
	if (tmpres.cols % step > 0) ++xsize;
	int ysize = tmpres.rows / step;
	if (tmpres.rows % step > 0) ++ysize;
	Mat dest = Mat::zeros(ysize, xsize, CV_64FC3);

	for (int i = 0; i < tmpres.rows; i++) 
	{
		for (int j = 0; j < tmpres.cols; j++) 
		{
			if (i % step > 0 || j % step > 0) continue;
			for (int ch = 0; ch < 3; ++ch) 
			{
				dest.AT3D(i / step, j / step)[ch] = tmpres.AT3D(i, j)[ch];
			}
			locat.push_back(tmplocat[i * tmpres.cols + j]);
		}
	}
	tmplocat.clear();
	std::vector<std::vector<Point> >().swap(tmplocat);
	return dest;
}

Mat PoolingOverlapTest(const Mat &matrix, Size2i windowSize, int step,
	int poolingMethod) 
{
	Mat tmpres = Mat::zeros(matrix.rows - windowSize.height + 1, matrix.cols - windowSize.width + 1, 
		CV_64FC3);
	std::vector<std::vector<Point> > tmplocat;
	for (int i = 0; i < matrix.rows - windowSize.height + 1; ++i) 
	{
		for (int j = 0; j < matrix.cols - windowSize.width + 1; ++j) 
		{
			Mat tmp;
			matrix(Rect(j, i, windowSize.width, windowSize.height)).copyTo(tmp);
			Scalar val = Scalar(0.0, 0.0, 0.0);
			if (POOLMAX == poolingMethod) 
			{
				Scalar minVal = Scalar(0.0, 0.0, 0.0);
				Scalar maxVal = Scalar(0.0, 0.0, 0.0);
				std::vector<Point> minLoc;
				std::vector<Point> maxLoc;
				minMaxLoc(tmp, minVal, maxVal, minLoc, maxLoc);
				val = maxVal;
			}
			else if(POOLSTOCHASTIC == poolingMethod) 
			{

				Scalar recip_sumval = sum(tmp);
				divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
				Mat prob = tmp.mul(recip_sumval);
				int ran = rand() % (tmp.rows * tmp.cols);
				std::vector<Point> loc = findLocCh(prob, ran);
				for (int ch = 0; ch < loc.size(); ch++) {
					val[ch] = tmp.AT3D(loc[ch].y, loc[ch].x)[ch];
				}
			}
			tmpres.AT3D(i, j) = Scalar2Vec3d(val);
		}
	}
	int xsize = tmpres.cols / step;
	if (tmpres.cols % step > 0) ++xsize;
	int ysize = tmpres.rows / step;
	if (tmpres.rows % step > 0) ++ysize;
	Mat dest = Mat::zeros(ysize, xsize, CV_64FC3);

	for (int i = 0; i < tmpres.rows; i++) {
		for (int j = 0; j < tmpres.cols; j++) {
			if (i % step > 0 || j % step > 0) continue;
			for (int ch = 0; ch < 3; ++ch) {
				dest.AT3D(i / step, j / step)[ch] = tmpres.AT3D(i, j)[ch];
			}
		}
	}
	return dest;
}

// Max pooling and stochastic pooling supported
Mat revPoolingOverlap(const Mat &matrix, Size2i windowSize, int step,
	int poolingMethod, std::vector<std::vector<Point> > &locat, Size2i upSize) 
{
	Mat res;
	if (windowSize.height == 1 && windowSize.width == 1 && step == 1) 
	{
		matrix.copyTo(res);
		return res;
	}
	res = Mat::zeros(upSize, CV_64FC3);
	for (int i = 0; i < matrix.rows; i++) 
	{
		for (int j = 0; j < matrix.cols; j++) 
		{
			for (int ch = 0; ch < 3; ch++) 
			{
				res.AT3D(locat[i * matrix.cols + j][ch].y, locat[i * matrix.cols + j][ch].x)[ch] += 
					matrix.AT3D(i, j)[ch];
			}
		}
	}
	return res;
}

Mat revPooling(const Mat &matrix, int step, int poolingMethod,
	std::vector<std::vector<Point> > &locat, Size2i upSize) 
{
	Mat res;
	if (step == 1)
	{
		matrix.copyTo(res);
		return res;
	}
	if (POOLMEAN == poolingMethod) 
	{

		Mat one = cv::Mat(step, step, CV_64FC3, Scalar(1.0, 1.0, 1.0));
		cout << matrix.size() << ",     " << one.size() << endl;
		res = kroneckerProduct(matrix, one);
		divide(res, Scalar(step * step, step * step, step * step), res);
	}
	else if(POOLMAX == poolingMethod || POOLSTOCHASTIC == poolingMethod) 
	{
		res = Mat::zeros(matrix.rows * step, matrix.cols * step, CV_64FC3);
		for (int i = 0; i < matrix.rows; i++) {
			for (int j = 0; j < matrix.cols; j++) {
				for (int ch = 0; ch < 3; ch++) {
					res.AT3D(locat[i * matrix.cols + j][ch].y, locat[i * matrix.cols + j][ch].x)[ch] 
						= matrix.AT3D(i, j)[ch];
				}
			}
		}
	}
	copyMakeBorder(res, res, 0, upSize.height - res.rows, 0, upSize.width - res.cols, 
		BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0));
	return res;
}


