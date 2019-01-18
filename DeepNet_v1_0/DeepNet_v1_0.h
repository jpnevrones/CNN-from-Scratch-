/*	DeepNet_v1_0.h  : Header file - include file for standard system include files, or project
*	specific include files that are used frequently, but are changed infrequently
*	version 0.1.1
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 03/31/2017 --    JP v0.1.1
*   - Updates to support overall network and layer change  -- 04/05/2017 --    JP v0.1.2
*	- Adding DeepNet parameter  -- 04/11/2017 --    JP v0.1.3
*
*/

#pragma once
//#define _SCL_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unordered_map>
#include "targetver.h"
#include <stdio.h>
#include <tchar.h>


// TODO: reference additional headers your program requires here

#define LRALPHA 100.0
#define ATD at<double>
#define AT3D at<cv::Vec3d>

// Conv2 parameter
#define CONVFULL 0
#define CONVSAME 1
#define CONVVALID 2
// Pooling methods
#define POOLMAX 0
#define POOLMEAN 1 
#define POOLSTOCHASTIC 2
//Non linear activation function
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define LEAKYRELU 3

#include <opencv2\core\utility.hpp>
#include <opencv2\core\ocl.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

#include "utilities.h"
#include "ioProcess.h"
#include "network.h"
#include "layer.h"
#include "channelThree.h"

using namespace std;
using namespace cv;




extern bool isGradientTrue;
extern bool enableNetLog;
extern int trainingEpochs;
extern int iterationPerEpoch;
extern double lrateweight;
extern double lratebias;

extern double momtumWeightInit;
extern double momtumSecDevInit;
extern double momtumWeightAdj;
extern double momtumSecDevAdj;
