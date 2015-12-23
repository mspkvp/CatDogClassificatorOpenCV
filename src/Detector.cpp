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

Ptr<SiftDescriptorExtractor> SiftExtractor::create()
{
	return new SiftDescriptorExtractor();
}

BOWKMeansTrainer BagOfWords::create(Mat descriptors, int dict_size) {
	int dictionarySize = dict_size; // the number of bags
	TermCriteria tc(CV_TERMCRIT_ITER, 5000, 0.00001); // define Term Criteria
	int retries = 1; // retries number
	int flags = KMEANS_PP_CENTERS; // necessary flags

	// create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	return bowTrainer;
}

void BagOfWords::saveToFile(Mat dictionary, String fileName) {
	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}

Ptr<DescriptorMatcher> Matcher::create() {
	return new FlannBasedMatcher();
}
