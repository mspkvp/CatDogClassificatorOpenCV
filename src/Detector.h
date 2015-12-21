#pragma once

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class SurfDetector {
public:
	static vector<KeyPoint> ExtractKeyPoints(Mat img, int HessianThreshold);
};