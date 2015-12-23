#include <iostream>
#include <fstream>
#include <map>
#include <opencv2/ml.hpp>
#include "opencv2/opencv.hpp"
#include "Detector.h"
#include <vector>
#include <direct.h>
#define GetCurrentDir _getcwd

String getCurrentPath() {
	char cCurrentPath[FILENAME_MAX];

	if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath))){
		return "";
	}

	cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */
	return cCurrentPath;
}

using namespace cv::ml;

int main(int argc, const char *argv[]) {

	string dataset_dir = getCurrentPath() + "\\train\\"; // argv[1];
	Mat training_descriptors;

	cout << "#### Building Vocabulary" << endl;
	// Vocabulary Building
	// read, detect and extract all features and descriptors from the set
	String animal = "cat";
	int animalCount = 12500;
	vector<vector<Mat>> animalImgs(2, vector<Mat>());
	
	for (int a = 0; a < 2; a++) {
		int counter = 0;
		cout << "\tProcessing " << animal << "s..." << endl;
		for (int i = 0; i < animalCount; i += 1250) {

			Mat img = imread(String(dataset_dir + animal + "." + to_string(i) + ".jpg"), IMREAD_GRAYSCALE);
			animalImgs[a].push_back(img);
			vector<KeyPoint> keypoints = SiftExtractor::ExtractKeyPoints(img);
			Mat descriptors = SiftExtractor::ExtractDescriptors(img, keypoints);

			training_descriptors.push_back(descriptors);
			cout << "\t\tProcessed " << i << "/" << animalCount << "\n";
		}
		animal = "dog";
	}

	cout << "## Writing Vocabulary to File" << endl;
	// Save Vocabulary to file
	FileStorage fs("training_descriptors.yml", FileStorage::WRITE);
	fs << "training_descriptors" << training_descriptors;
	fs.release();

	cout << "#### Creating Bag of Words" << endl;
	// Create bag of words
	BOWKMeansTrainer bowTrainer = BagOfWords::create(training_descriptors, 200);
	//cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(training_descriptors);
	BagOfWords::saveToFile(dictionary, "dictionary_bow.yml");
	
	BOWImgDescriptorExtractor bowide(SiftExtractor::create(), Matcher::create());
	bowide.setVocabulary(dictionary);

	cout << "#### SVM Training" << endl;
	map<string, Mat> classes_training_data; 
	classes_training_data.clear();

	/** SVM TRAINING **/
	Mat response_hist;
	animal = "cat";
	for (int a = 0; a < 2; a++) {
		cout << "\tProcessing " << animal << "s..." << endl;
		int counter = 0;
		for (int i = 0; i < animalCount; i += 1250) {
			
			cout << String(dataset_dir + animal + "." + to_string(i) + ".jpg") << "\n";

			Mat img = imread(String(dataset_dir + animal + "." + to_string(i) + ".jpg"), IMREAD_GRAYSCALE);

			vector<KeyPoint> keypoints2 = SiftExtractor::ExtractKeyPoints(img);
			Mat descriptors = SiftExtractor::ExtractDescriptors(img, keypoints2);

			bowide.compute(descriptors, response_hist);
			
			if (classes_training_data.count(animal) == 0) { //not yet created...
				classes_training_data[animal].create(0, response_hist.cols, response_hist.type());
			}
			classes_training_data[animal].push_back(response_hist);

			cout << "\t\tProcessed " << i << "/" << animalCount << "\n";
		}
		animal = "dog";
	}
	Ptr<ml::SVM> svm = ml::SVM::create();

	cout << "#### SVM 1vsAll Training" << endl;
	//train 1-vs-all SVMs
	map<string, Ptr<ml::SVM>> classes_classifiers;
	for (map<string, Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		string class_ = (*it).first;
		cout << "\tTraining Class: " << class_ << ".." << endl;

		Mat samples(0, response_hist.cols, response_hist.type());
		
		Mat labels(0, 1, CV_32FC1);

		//copy class samples and label
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);

		//copy rest samples and label
		for (map<string, Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			string not_class_ = (*it1).first;
			if (not_class_[0] == class_[0]) continue;
			samples.push_back(classes_training_data[not_class_]);
			class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
			labels.push_back(class_label);
		}

		Mat samples_32f; samples.convertTo(samples_32f, CV_32F);
		classes_classifiers[class_] = ml::SVM::create();
		classes_classifiers[class_]->train(samples_32f, ml::ROW_SAMPLE, labels);
	}

	cout << "#### Testing !" << endl;
	String test_set_dir = "./test1/";
	for (int i = 1; i < animalCount; i += 50) {
		Mat img = imread(String(dataset_dir + animal + "." + to_string(i) + ".jpg"), IMREAD_GRAYSCALE);
		vector<KeyPoint> keypoints = SiftExtractor::ExtractKeyPoints(img);
		Mat descriptors = SiftExtractor::ExtractDescriptors(img, keypoints);

		bowide.compute(img, keypoints, response_hist);

		for (map<string, Ptr<ml::SVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second->predict(response_hist);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}

		cout << "\tProcessed " << i << "/" << animalCount << "\n";
	}

	return 0;
}
