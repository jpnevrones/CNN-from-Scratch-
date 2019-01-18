#pragma once
#include "DeepNet_v1_0.h"

using namespace std;
using namespace cv;
/*Three Channel support*/



typedef Mat(*func)(const Mat&);
typedef Mat(*func2)(const Mat&, const Mat&);
typedef Mat(*func3)(const Mat&, const Mat&, int);
typedef Mat(*func4)(const Mat&, int);
typedef Mat(*func5)(const Mat&, const Mat&, int, int, int);
typedef Mat(*func6)(const Mat&, int, int);

Mat parallel(func, const Mat &);

Mat parallel(func2, const Mat&, const Mat&);

Mat parallel(func3, const Mat&, const Mat&, int);

Mat parallel(func4, const Mat&, int);

Mat parallel(func5, const Mat&, const Mat&, int, int, int);

Mat parallel(func6, const Mat&, int, int);