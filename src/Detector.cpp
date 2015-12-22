#include "Detector.h"

vector<KeyPoint> SurfExtractor::ExtractKeyPoints(Mat img, int hessianThreshold) {	
	Ptr<SURF> detector = SURF::create(hessianThreshold);
	
	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	return keypoints;
}

Mat SurfExtractor::ExtractDescriptors(Mat img, vector<KeyPoint> keypoints, int hessianThreshold) {
	Ptr<SURF> detector = SURF::create(hessianThreshold);

	Mat descriptors;
	detector->compute(img, keypoints, descriptors);
	return descriptors;
}

vector<KeyPoint> SiftExtractor::ExtractKeyPoints(Mat img) {
	Ptr<SIFT> detector = SIFT::create();

	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	return keypoints;
}

Mat SiftExtractor::ExtractDescriptors(Mat img, vector<KeyPoint> keypoints) {
	Ptr<SIFT> detector = SIFT::create();

	Mat descriptors;
	detector->compute(img, keypoints, descriptors);

	return descriptors;
}

Mat BagOfWords::create(Mat features) {
	int dictionarySize = 200; // the number of bags
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001); // define Term Criteria
	int retries = 1; // retries number
	int flags = KMEANS_PP_CENTERS; // necessary flags

	// create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	return bowTrainer.cluster(features);
}

void BagOfWords::saveToFile(Mat dictionary, String fileName) {
	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}
