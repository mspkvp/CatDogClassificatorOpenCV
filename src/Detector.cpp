#include "Detector.h"

vector<KeyPoint> SurfDetector::ExtractKeyPoints(Mat img, int HessianThreshold)
{	
	Ptr<SURF> detector = SURF::create(HessianThreshold);
	
	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	return keypoints;
}

vector<KeyPoint> SiftExtractor::ExtractKeyPoints(Mat img)
{
	Ptr<SIFT> detector = SIFT::create();

	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	return keypoints;
}

Mat SiftExtractor::ExtractDescriptors(Mat img, vector<KeyPoint> keypoints){
	Ptr<SIFT> detector = SIFT::create();

	Mat descriptors;
	detector->compute(img, keypoints, descriptors);

	return descriptors;
}
