#include <iostream>
#include "Detector.h"

int main(int argc, const char *argv[]) {

	/*
	// Testing SURF #################################################
	Mat img_1 = imread("./train/cat.2.jpg", IMREAD_GRAYSCALE);
	Mat img_2 = imread("./train/cat.1.jpg", IMREAD_GRAYSCALE);
	vector<KeyPoint> keypoints_1 = SurfDetector::ExtractKeyPoints(img_1, 500), 
					keypoints_2 = SurfDetector::ExtractKeyPoints(img_2, 500);

	vector<KeyPoint> keypoints_1 = SiftDetector::ExtractKeyPoints(img_1),
					keypoints_2 = SiftDetector::ExtractKeyPoints(img_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);
	waitKey(0);
	// Testing SURF end #############################################
	return 0;*/

	string dataset_dir = "./train/"; // argv[1];

	Mat featuresUnclustered_cats;
	/* CATS */
	// read, detect and extract all features and descriptors from the set
	for (int i = 0; i < 12500; i++){
		Mat img = imread(String(dataset_dir+"cat."+i+".jpg"), IMREAD_GRAYSCALE);

		vector<Keypoint> keypoints = SiftExtractor::ExtractKeyPoints(img);
		Mat descriptors = SiftExtractor::ExtractDescriptors(img, keypoints);

		featuresUnclustered_cats.push_back(descriptors);
	}

	return 0;
}
