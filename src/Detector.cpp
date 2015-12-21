#include "Detector.h"

vector<KeyPoint> SurfDetector::ExtractKeyPoints(Mat img, int HessianThreshold)
{	
	Ptr<SURF> detector = SURF::create(HessianThreshold);
	
	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	return keypoints;
}
