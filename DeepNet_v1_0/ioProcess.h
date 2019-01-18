/*	ioProcess.cpp : source file - contains the matrix based compuation utilities
*	and other set of utilities used by the core Framework
*	version 0.1.0
*	Developers : JP - Jithin Pradeep(jxp161430)
*	Last Update:
*	Initial draft -- 04/17/2017 --    JP v0.1.1
*
*/
using namespace std;
using namespace cv;



void getBatch(string, vector<Mat>&, Mat&);
void getCIFAR10(vector<Mat>&, Mat&, vector<Mat>&,  Mat&);
Mat concat(const vector<Mat> &);
void preProcessing(vector<Mat>&, vector<Mat>&);

/*void save2XML(std::vector<layer*>&, string, string);*/