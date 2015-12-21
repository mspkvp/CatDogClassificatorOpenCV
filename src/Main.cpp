#include <iostream>
#include "Detector.h"

int main(int argc, const char *argv[]) {

	if (argc < 3)
		return -1;

	// Testing SURF #################################################
	Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
	Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

	vector<KeyPoint> keypoints_1 = SurfDetector::ExtractKeyPoints(img_1, 500), 
					keypoints_2 = SurfDetector::ExtractKeyPoints(img_2, 500);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	waitKey(0);
	// Testing SURF end #############################################
	return 0;
}
