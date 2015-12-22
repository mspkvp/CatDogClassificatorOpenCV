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

class SurfExtractor {
public:
	static vector<KeyPoint> ExtractKeyPoints(Mat img, int hessianThreshold);
	static Mat ExtractDescriptors(Mat img, vector<KeyPoint> keypoints, int hessianThreshold);
};

class SiftExtractor {
public:
	static vector<KeyPoint> ExtractKeyPoints(Mat img);
	static Mat ExtractDescriptors(Mat img, vector<KeyPoint> keypoints);

};

class BagOfWords {
public:
	static Mat create(Mat features);
	static void saveToFile(Mat dictionary, String fileName);
};

class Matcher {
public:

};