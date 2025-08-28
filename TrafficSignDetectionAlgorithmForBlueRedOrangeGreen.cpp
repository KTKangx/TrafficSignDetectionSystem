#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<iostream>
#include	<stdlib.h>
#include	<string>
#include	"Supp.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

void Preprocess(Mat& srcI, Mat& hsv, Mat& Large, Mat win[], Mat legend[], int noOfImagePerCol, int noOfImagePerRow, vector<Mat>& hsvChannels, Mat& hsvEnhanced, Mat& enhancedImage)
{
	cv::resize(srcI, srcI, cv::Size(135, 135), 0, 0, cv::INTER_LINEAR);
	createWindowPartition(srcI, Large, win, legend, noOfImagePerCol, noOfImagePerRow);
	cvtColor(srcI, hsv, COLOR_BGR2HSV);

	split(hsv, hsvChannels);

	double saturationFactor = 1.1, brightnessFactor = 1.7; // Adjust this factor to increase or decrease saturation

	hsvChannels[1] *= saturationFactor;
	hsvChannels[2] *= brightnessFactor;

	// Clip the values to stay within valid range [0, 255]
	hsvChannels[1] = min(hsvChannels[1], 255);
	merge(hsvChannels, hsvEnhanced);

	cvtColor(hsvEnhanced, enhancedImage, COLOR_HSV2BGR);

}


Mat redMask(Mat hsvEnhanced)
{
	Mat			 redMask1, redMask2, result;

	//split(hsvEnhanced, threeImages); // split image into its BGR components
	//// Below get red dominant regions / points
	//redMask1 = (threeImages[0] * 1.5 < threeImages[2]) &
	//	(threeImages[1] * 1.5 < threeImages[2]);
	////cvtColor(redMask1, result, COLOR_GRAY2BGR); // show result of red color
	Scalar RedLower(0, 90, 50);
	Scalar RedLower2(8, 255, 255);
	Scalar RedUpper(165, 110, 50);
	Scalar RedUpper2(179, 255, 255);

	inRange(hsvEnhanced, RedLower, RedLower2, redMask1);	//create mask for Red traffic sign
	inRange(hsvEnhanced, RedUpper, RedUpper2, redMask2);

	bitwise_or(redMask1, redMask2, result);

	return result;
}

Mat OrangeMask(Mat hsvEnhanced)
{
	Mat mask;
	Scalar OrangeLower(10, 110, 110);
	Scalar OrangeUpper(30, 255, 255);

	inRange(hsvEnhanced, OrangeLower, OrangeUpper, mask);	//create mask for Orange traffic sign

	return mask;
}

Mat YellowMask(Mat hsvEnhanced)
{
	Mat mask;
	Scalar YellowLower(27, 120, 100);
	Scalar YellowUpper(38, 255, 255);
	inRange(hsvEnhanced, YellowLower, YellowUpper, mask);	//create mask for yellow traffic sign

	return mask;
}

Mat BlueMask(Mat hsvEnhanced)
{
	Mat mask;
	Scalar BlueLower(90, 110, 70);
	Scalar BlueUpper(125, 255, 255);

	inRange(hsvEnhanced, BlueLower, BlueUpper, mask);	//create mask for Orange traffic sign

	return mask;
}

void ContourDrawingAndFilling(Mat& canvasGray, vector<vector<Point> >	contours, Mat& canvasColor, vector<Scalar>	colors, Point2i	center, Mat win[], Mat win3[], Mat srcI, vector<Mat>& LongestFill)
{
	int max = 0, index = 0;


	for (int i = 0; i < contours.size(); i++) { // We could have more than one sign in image

		//canvasGray = 0;
		canvasColor = Scalar(0, 0, 0);

		if (max < contours[i].size()) { // Find the longest contour as sign boundary
			max = contours[i].size();
			index = i;
		}

		drawContours(canvasColor, contours, i, colors[i]); // draw Color boundaries
		canvasColor.copyTo(win[2]);

		canvasGray = 0;

		drawContours(canvasGray, contours, index, 255);	//draw the longest contour
		cvtColor(canvasGray, win[4], COLOR_GRAY2BGR);

		Moments M = moments(canvasGray);
		center.x = M.m10 / M.m00;
		center.y = M.m01 / M.m00;

		floodFill(canvasGray, center, 255);
		cvtColor(canvasGray, win[5], COLOR_GRAY2BGR);

		//If longest contour fill with size of pixels > 3000 only considered containing traffic sign
		if (countNonZero(canvasGray) > 3000 && countNonZero(canvasGray) < 11001)
		{
			Mat temp, Longest;
			cvtColor(canvasGray, temp, COLOR_GRAY2BGR);

			//Extract the captured sign
			Longest = srcI & temp;
			Longest.copyTo(win[6]);
			Longest.copyTo(win3[4]);

			cvtColor(temp, temp, COLOR_BGR2GRAY);

			//copy longest fill image to vector 
			LongestFill.push_back(temp);
		}

	}

}

void ResultPrint(Mat win[], Mat& mask4, Mat& mask3, Mat& mask2, Mat& mask, Mat srcI)
{
	Mat Red, Yellow, Orange, Blue;

	cvtColor(mask4, mask4, COLOR_GRAY2BGR);		//	Blue_mask
	cvtColor(mask3, mask3, COLOR_GRAY2BGR);		//	Yellow_mask
	cvtColor(mask2, mask2, COLOR_GRAY2BGR);		//	Orange_mask
	cvtColor(mask, mask, COLOR_GRAY2BGR);		//	Red_mask

	Blue = srcI & mask4;	//Blue_mask result
	Yellow = srcI & mask3;	//yellow mask result
	Orange = srcI & mask2;	//orange mask result
	Red = srcI & mask;		//red mask result

	Red.copyTo(win[0]);
	Yellow.copyTo(win[1]);
	Orange.copyTo(win[2]);
	Blue.copyTo(win[3]);
}

void Morphology(Mat& Result, Mat PreprocessImage)	//Not Used
{
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(PreprocessImage, Result, MORPH_OPEN, kernel);

}

vector<float> extractBGRFeatures(const Mat& mask, const Mat& img) {
	vector<float> histogram;
	Mat hist;
	int histSize[] = { 50, 50, 50 };
	float bRanges[] = { 0, 256 };
	float gRanges[] = { 0, 256 };
	float rRanges[] = { 0, 256 };
	const float* ranges[] = { bRanges, gRanges, rRanges };
	int channels[] = { 0, 1, 2 };

	calcHist(&img, 1, channels, mask, hist, 3, histSize, ranges, true, false);
	normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

	histogram.insert(histogram.end(), hist.begin<float>(), hist.end<float>());

	return histogram;
}

void ImageColor(Mat finalImage, vector<string>& color)
{
	Mat hsv;
	vector<Mat> hsvChannels;
	double pixels[4], percentage[4], Percentage;
	int index;
	string Color[4]{ "red","orange","yellow","blue" };

	cvtColor(finalImage, hsv, COLOR_BGR2HSV);

	split(hsv, hsvChannels);

	double saturationFactor = 1.1, brightnessFactor = 1.7; // Adjust this factor to increase or decrease saturation

	hsvChannels[1] *= saturationFactor;
	hsvChannels[2] *= brightnessFactor;

	// Clip the values to stay within valid range [0, 255]
	hsvChannels[1] = min(hsvChannels[1], 255);
	merge(hsvChannels, finalImage);

	//cvtColor(finalImage, finalImage, COLOR_HSV2BGR);

	Mat mask = redMask(finalImage);
	pixels[0] = countNonZero(mask);
	mask = OrangeMask(finalImage);
	pixels[1] = countNonZero(mask);
	mask = YellowMask(finalImage);
	pixels[2] = countNonZero(mask);
	mask = BlueMask(finalImage);
	pixels[3] = countNonZero(mask);

	for (int i = 0; i < 4; i++)
	{
		percentage[i] = pixels[i] / (finalImage.cols * finalImage.rows) * 100;
		cout << endl << Color[i] << " percentage: " << percentage[i] << endl << endl;
	}
	Percentage = percentage[0];
	index = 0;

	for (int i = 0; i < 3; i++)
		if (percentage[i + 1] > Percentage)
		{
			Percentage = percentage[i + 1];
			index = i + 1;
		}
	color.push_back(Color[index]);
}

// feature extraction number of vertices
int extractVertices(const Mat& image) {

	/* //this is commend out for testing purpose because cannot run if in my fyp1 code
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, gray, 127, 255, THRESH_BINARY);
	*/
	vector<vector<Point>> contours;
	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); // image is replaced from gray for testing

	int longestContourIndex = -1;
	double maxContourLength = 0;

	// find logest contour
	for (size_t i = 0; i < contours.size(); i++) {
		double contourLength = arcLength(contours[i], true);
		if (contourLength > maxContourLength) {
			maxContourLength = contourLength;
			longestContourIndex = (int)i;
		}
	}


	cout << "Longest contour index: " << longestContourIndex << endl;


	if (longestContourIndex != -1) {
		vector<Point> approx;
		approxPolyDP(contours[longestContourIndex], approx, arcLength(contours[longestContourIndex], true) * 0.04, true);
		cout << "Approx vertices: " << approx.size() << endl;
		return approx.size();
	}

	return 0;
}


vector<double> extractHuMoments(const Mat& img) {
	/* // this is commend out for testing as it cannot run in my fyp1 code
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	*/

	vector<vector<Point>> contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); // image is replaced from gray for testing

	int longestContourIndex = -1;
	double maxContourLength = 0;

	// find logest contour
	for (size_t i = 0; i < contours.size(); i++) {
		double contourLength = arcLength(contours[i], true);
		if (contourLength > maxContourLength) {
			maxContourLength = contourLength;
			longestContourIndex = (int)i;
		}
	}


	cout << "Longest contour index: " << longestContourIndex << endl;
	//test

	if (countNonZero(img) == 0) {
		cout << "Error: Image contains no non-zero pixels in extractHuMoments!" << endl;
		return vector<double>(7, 0);
	}

	if (longestContourIndex == -1) {
		cout << "Error: longest contour index == -1!" << endl;
		return vector<double>(7, 0);
	}


	Moments m = moments(contours[longestContourIndex]); //img is replaced from gray for testing
	//test


	if (m.m00 == 0) {
		cout << "Warning: Moment calculation failed (m.m00 is zero)" << endl;
		return vector<double>(7, 0);
	}

	double hu[7];
	HuMoments(m, hu);

	// Debug: Print Hu moments
	cout << "Calculated Hu Moments: ";
	for (int i = 0; i < 7; i++) {
		cout << hu[i] << " ";
	}
	cout << endl;


	vector<double> huFeatures(hu, hu + 7);
	return huFeatures;
}

// Compare hu moment by shape in labelDetermine function
bool compareHuMoments(const double hu[7], const double shapeHu[7], double threshold = 0.001) {
	for (int i = 0; i < 7; i++) {
		if (fabs(hu[i] - shapeHu[i]) > threshold) {
			return false;
		}
	}
	return true;
}


int labelDetermine(int testingvertices, const vector<double>& huMomentsTest, const string& labelColourType) {
	// Reference HU moments for shapes
	double triangleHu[7] = { 0.188609, 8.20623e-05, 0.00386594, 3.4332e-06, 1.67763e-10, 2.73115e-09, 3.58186e-10 };
	//double squareHu[7] = { 0.167653, 0.000135753, 1.31157e-05, 4.39247e-07, 9.77265e-13, 4.88695e-09, 3.95573e-13 };
	double squareHu[7] = { 0.28379, 1.03082e-09, 2.02043e-13, 1.69373e-06, -9.90801e-16, -5.43794e-11, 0 }; //new 
	double circleHu[7] = { 0.159456, 9.14777e-05, 2.05338e-07,  1.83226e-10, -5.99227e-19, -8.65957e-13, 9.50793e-19 };
	double octagonHu[7] = { 0.159655, 5.27376e-05, 4.63002e-07, 3.82563e-10, 1.04382e-18, -2.58676e-12, 4.98335e-18 };

	// Check if all Hu moments are zero
	bool allZero = true;
	for (double val : huMomentsTest) {
		if (val != 0) {
			allZero = false;
			break;
		}
	}

	if (allZero) {
		return 10; // Others (All Hu moments are zero)
	}

	// Label based on colour type and vertices
	if (labelColourType == "red") {
		if (testingvertices == 3 || compareHuMoments(huMomentsTest.data(), triangleHu)) {
			return 0; // Red triangle
		}
		else if (testingvertices == 4 || compareHuMoments(huMomentsTest.data(), squareHu)) {
			return 1; // Red square
		}
		else if (testingvertices == 8 && compareHuMoments(huMomentsTest.data(), octagonHu)) {
			return 2; // Red octagon
		}
		else if (testingvertices > 4 || compareHuMoments(huMomentsTest.data(), circleHu)) {
			return 3; // Red circle
		}
	}
	else if (labelColourType == "yellow") {
		if (testingvertices == 3 || compareHuMoments(huMomentsTest.data(), triangleHu)) {
			return 4; // Yellow triangle
		}
		else if (testingvertices == 4 || compareHuMoments(huMomentsTest.data(), squareHu)) {
			return 5; // Yellow square
		}
		else if (testingvertices > 4 || compareHuMoments(huMomentsTest.data(), circleHu)) {
			return 6; // Yellow circle
		}
	}
	else if (labelColourType == "blue") {
		if (testingvertices == 3 || compareHuMoments(huMomentsTest.data(), triangleHu)) {
			return 7; // Blue triangle
		}
		else if (testingvertices == 4 || compareHuMoments(huMomentsTest.data(), squareHu)) {
			return 8; // Blue square
		}
		else if (testingvertices > 4 || compareHuMoments(huMomentsTest.data(), circleHu)) {
			return 9; // Blue circle
		}
	}

	return -1; // Return -1 if no match (this shouldn't happen if logic is correct)
}


// SVM Training
Ptr<SVM> trainSVM(const vector<vector<float>>& featureVectors, const vector<int>& labels) {
	Mat trainData = Mat(featureVectors.size(), featureVectors[0].size(), CV_32F);
	Mat trainLabels = Mat(labels.size(), 1, CV_32S);


	for (int i = 0; i < featureVectors.size(); i++) {
		for (int j = 0; j < featureVectors[i].size(); j++) {
			trainData.at<float>(i, j) = static_cast<float>(featureVectors[i][j]);
		}
		trainLabels.at<int>(i, 0) = labels[i];
	}

	Ptr<SVM> svm = SVM::create();
	svm->setKernel(SVM::LINEAR);
	svm->setType(SVM::C_SVC);
	svm->setC(1.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	return svm;
}


// k-NN Training
Ptr<KNearest> trainKNN(const vector<vector<float>>& featureVectors, const vector<int>& labels) {
	Mat trainData = Mat(featureVectors.size(), featureVectors[0].size(), CV_32F);
	Mat trainLabels = Mat(labels.size(), 1, CV_32S);


	for (int i = 0; i < featureVectors.size(); i++) {
		for (int j = 0; j < featureVectors[i].size(); j++) {
			trainData.at<float>(i, j) = static_cast<float>(featureVectors[i][j]);
		}
		trainLabels.at<int>(i, 0) = labels[i];
	}


	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(3); // Set the number of neighbors
	knn->train(trainData, ROW_SAMPLE, trainLabels);
	return knn;
}

//random forest
Ptr<RTrees> trainForest(const vector<vector<float>>& featureVectors, const vector<int>& labels) {
	Mat trainData = Mat(featureVectors.size(), featureVectors[0].size(), CV_32F);
	Mat trainLabels = Mat(labels.size(), 1, CV_32S);


	for (int i = 0; i < featureVectors.size(); i++) {
		for (int j = 0; j < featureVectors[i].size(); j++) {
			trainData.at<float>(i, j) = static_cast<float>(featureVectors[i][j]);
		}
		trainLabels.at<int>(i, 0) = labels[i];
	}

	// Create and configure the random forest model
	Ptr<ml::RTrees> randomForest = ml::RTrees::create();
	randomForest->setMaxDepth(7);         // Set maximum depth of the tree
	randomForest->setMaxCategories(15);    // Max number of categories (useful for categorical variables)
	randomForest->setPriors(Mat());        // Priors of each class
	randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.01));

	// Train the random forest
	randomForest->train(trainData, ml::ROW_SAMPLE, trainLabels);

	return randomForest;
}


// Split data into train and test
void splitData(const vector<vector<float>>& data, const vector<int>& labels, float trainRatio,
	vector<vector<float>>& trainData, vector<vector<float>>& testData, vector<int>& trainLabels, vector<int>& testLabels)
{

	vector<int> indices(data.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}

	// Shuffle the indices using seed for validation
	unsigned int seed = 52;
	srand(seed);
	random_shuffle(indices.begin(), indices.end());
	int trainSize = static_cast<int>(trainRatio * data.size());

	for (int i = 0; i < trainSize; ++i) {
		trainData.push_back(data[indices[i]]);
		trainLabels.push_back(labels[indices[i]]);
	}

	for (int i = trainSize; i < indices.size(); ++i) {
		testData.push_back(data[indices[i]]);
		testLabels.push_back(labels[indices[i]]);
	}
	//   cout << train
}


// Function to calculate accuracy
double calculateAccuracy(vector<int>& trueLabels, vector<int>& predictions) {
	int correct = 0;
	for (size_t i = 0; i < trueLabels.size(); i++) {
		if (trueLabels[i] == predictions[i]) correct++;
	}
	return (double)correct / trueLabels.size() * 100.0;
}

// Function to calculate Precision, Recall, and F1-Score
void calculateMetrics(vector<int>& trueLabels, vector<int>& predictions,
	double& precision, double& recall, double& f1Score) {

	int TP = 0, FP = 0, TN = 0, FN = 0;

	for (size_t i = 0; i < trueLabels.size(); i++) {
		if (predictions[i] == 1 && trueLabels[i] == 1) TP++;
		else if (predictions[i] == 1 && trueLabels[i] == 0) FP++;
		else if (predictions[i] == 0 && trueLabels[i] == 1) FN++;
		else if (predictions[i] == 0 && trueLabels[i] == 0) TN++;
	}

	// Avoid division by zero for precision
	if (TP + FP == 0) {
		precision = 0;
	}
	else {
		precision = TP / (double)(TP + FP);
	}

	// Avoid division by zero for recall
	if (TP + FN == 0) {
		recall = 0;
	}
	else {
		recall = TP / (double)(TP + FN);
	}

	// Avoid division by zero for F1-Score
	if (precision + recall == 0) {
		f1Score = 0;
	}
	else {
		f1Score = 2 * (precision * recall) / (precision + recall);
	}
}



int main()
{
	String		imgPattern("Inputs/Traffic signs/*.png");
	vector<string>	imageNames;
	int const	noOfImagePerCol = 5, noOfImagePerRow = 2;
	Mat threeImages[3], srcI, thresh, canvasColor, canvasGray, gaussian, hsv, test, blur;
	vector<Scalar>	colors;
	int			t1, t2, t3, t4, index, max = 0;// used to record down the longest contour
	RNG			rng(0);
	Point2i		center;


	//store feature and label
	vector<vector<float>> trainingData;
	vector<int> labels;

	Mat Result, Result_win[1 * 5], legend3[1 * 5];
	Mat Red_large, Orange_large, Yellow_large, Blue_large, win[4][noOfImagePerCol * noOfImagePerRow], legend[4][noOfImagePerCol * noOfImagePerRow];

	for (int i = 0; i < 300; i++) {
		for (;;) {
			t1 = rng.uniform(0, 255); // blue
			t2 = rng.uniform(0, 255); // green
			t3 = rng.uniform(0, 255); // red
			t4 = t1 + t2 + t3;
			// Below get random colors that is not dim
			if (t4 > 255) break;
		}
		colors.push_back(Scalar(t1, t2, t3));
	}

	cv::glob(imgPattern, imageNames, true);

	//cout << imageNames.size();
	for (int j = 0; j < imageNames.size(); j++)
	{
		srcI = imread(imageNames[j]);
		//cvtColor(srcI, srcI, COLOR_GRAY2BGR);

		if (srcI.empty()) { // found no such file?
			cout << "cannot open image for reading" << endl;
			return -1;
		}


		//create hsv channels to contain split hsv traits
		vector<Mat> hsvChannels;
		Mat hsvEnhanced, enhancedImage, Red_mask, Orange_mask, Yellow_mask, Blue_mask;

		//Used to find contour that contains Traffic Sign but not others
		vector<vector<Point> >	Contour, Contour2, FinalContour;
		//store all longestFill image
		vector<Mat>	LongestFill;
		//store shortlisted final image (Mask or LongestFill)
		vector<Mat>	CounterAndMask;

		//Preprocess picture and Create windows to display
		Preprocess(srcI, hsv, Red_large, win[0], legend[0], noOfImagePerCol, noOfImagePerRow, hsvChannels, hsvEnhanced, enhancedImage);

		Preprocess(srcI, hsv, Orange_large, win[1], legend[1], noOfImagePerCol, noOfImagePerRow, hsvChannels, hsvEnhanced, enhancedImage);

		Preprocess(srcI, hsv, Yellow_large, win[2], legend[2], noOfImagePerCol, noOfImagePerRow, hsvChannels, hsvEnhanced, enhancedImage);

		Preprocess(srcI, hsv, Blue_large, win[3], legend[3], noOfImagePerCol, noOfImagePerRow, hsvChannels, hsvEnhanced, enhancedImage);

		//Preprocess(srcI, hsv, newFrame, Mor_win, legend2, noOfImagePerCol, noOfImagePerRow, hsvChannels, hsvEnhanced, enhancedImage);

		Preprocess(srcI, hsv, Result, Result_win, legend3, 1, 5, hsvChannels, hsvEnhanced, enhancedImage);

		Red_mask = redMask(hsvEnhanced);
		Orange_mask = OrangeMask(hsvEnhanced);
		Yellow_mask = YellowMask(hsvEnhanced);
		Blue_mask = BlueMask(hsvEnhanced);

		//Vector stores all masked results (4 colors Masks results)
		vector<Mat> Masks{ Red_mask, Orange_mask, Yellow_mask, Blue_mask };

		ResultPrint(Result_win, Blue_mask, Yellow_mask, Orange_mask, Red_mask, srcI);

		vector<vector<Point>> contours;
		//create canvases for line drawing
		canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
		canvasGray.create(srcI.rows, srcI.cols, CV_8U);

		for (int i = 0; i < Masks.size(); i++)
		{
			findContours(Masks[i], contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			ContourDrawingAndFilling(canvasGray, contours, canvasColor, colors, center, win[i], Result_win, srcI, LongestFill);

			//Red
			putText(legend[0][0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][1], "", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][2], "Color contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][3], "Gray contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][4], "FillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][5], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][6], "LongFillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][7], "Red Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][8], "OrangeResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[0][9], "YellowResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			//Orange
			putText(legend[1][0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][1], "", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][2], "Color contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][3], "Gray contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][4], "FillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][5], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][6], "LongFillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][7], "Orange Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][8], "OrangeResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1][9], "YellowResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			//Yellow
			putText(legend[2][0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][1], "", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][2], "Color contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][3], "Gray contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][4], "FillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][5], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][6], "LongFillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][7], "Yellow Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][8], "OrangeResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[2][9], "YellowResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			//Blue
			putText(legend[3][0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][1], "", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][2], "Color contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][3], "Gray contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][4], "FillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][5], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][6], "LongFillContour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][7], "Blue Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][8], "OrangeResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[3][9], "YellowResult", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			//Results
			putText(legend3[0], "Red", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend3[1], "Yellow", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend3[2], "Orange", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend3[3], "Blue", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend3[4], "Longest", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			Mat Mask;

			srcI.copyTo(win[0][0]);
			srcI.copyTo(win[1][0]);
			srcI.copyTo(win[2][0]);
			srcI.copyTo(win[3][0]);

			Mask = Red_mask & srcI;
			Mask.copyTo(win[0][7]);

			Mask = Orange_mask & srcI;
			Mask.copyTo(win[1][7]);

			Mask = Yellow_mask & srcI;
			Mask.copyTo(win[2][7]);

			Mask = Blue_mask & srcI;
			Mask.copyTo(win[3][7]);

		}


		vector<string> color;

		//Processes all shortlisted Mask and LongestFill images to find the real traffic sign
		for (int i = 0; i < std::max(Masks.size(), LongestFill.size()); i++)
		{

			//4 Masks images filter Model
			if (i < Masks.size())
			{

				//image with at least 1500 pixel only possible to contain traffic sign
				if (countNonZero(Masks[i]) > 1500 && countNonZero(Masks[i]) < 11001)
				{
					findContours(Masks[i], Contour, RETR_EXTERNAL, CHAIN_APPROX_NONE);
					//filter all noise which are not traffic sign
					for (int y = 0; y < Contour.size(); y++)
					{
						Rect boundingRect = cv::boundingRect(Contour[y]);

						if (cv::contourArea(Contour[y]) > 1500 && cv::contourArea(Contour[y]) < 11001)
						{
							//Traffic sign shape usually close to square 
							if (boundingRect.x < boundingRect.y * 1.2 || boundingRect.x * 1.2 > boundingRect.y)
							{
								//Shortlisted images are pushed into vector
								CounterAndMask.push_back(Masks[i]);
								FinalContour.push_back(Contour[y]);
							}
						}
					}
				}
			}

			//LongestFill images filter Model
			if (i < LongestFill.size())
				//	cout << endl <<"pixel: " << countNonZero(LongestFill[i]) << endl;

				if (LongestFill.size() != 0)
				{
					//image with at least 1500 pixel only possible to contain traffic sign
					if (countNonZero(LongestFill[0]) > 1500 && countNonZero(LongestFill[0]) < 11001)
					{
						findContours(LongestFill[0], Contour2, RETR_EXTERNAL, CHAIN_APPROX_NONE);

						for (int z = 0; z < Contour2.size(); z++)
						{
							Rect boundingRect2 = cv::boundingRect(Contour2[z]);

							if (cv::contourArea(Contour2[z]) > 1500 && cv::contourArea(Contour2[z]) < 11001)
							{
								//Traffic sign shape usually close to square 
								if (boundingRect2.x < boundingRect2.y * 1.2 || boundingRect2.x * 1.2 > boundingRect2.y)
								{
									//Shortlisted images are pushed into vector
									CounterAndMask.push_back(LongestFill[0]);
									FinalContour.push_back(Contour2[z]);
								}
							}
						}
					}
					LongestFill.erase(LongestFill.begin() + 0);	//Erase the image from vector Whether shortlisted or not (Not repeatedly analyse the same image)
				}

			//	cout << endl << "total pic: " << CounterAndMask.size() << endl;
			if (CounterAndMask.size() > 1)
			{
				int value;

				//Pick only one image from all shortlisted image for final Showcase
				for (int i = 0; i < (CounterAndMask.size()); i++)
				{
					value = countNonZero(CounterAndMask[0]);

					//if Image 1 > Image 2, then Image 2 is deleted
					if (value >= countNonZero(CounterAndMask[1]))
					{
						CounterAndMask.erase(CounterAndMask.begin() + 1);
						FinalContour.erase(FinalContour.begin() + 1);
					}

					//if Image 2 > Image 1, then Image 1 is deleted
					else
					{
						value = countNonZero(CounterAndMask[1]);
						CounterAndMask.erase(CounterAndMask.begin() + 0);
						FinalContour.erase(FinalContour.begin() + 0);
					}
				}
			}
			//	cout << endl << "contour:" << FinalContour.size() << endl;
		}


		Mat GreenBox, temp;
		srcI.copyTo(GreenBox);

		if (FinalContour.size() > 0)
		{
			//Draw Green box around detected components (Traffic sign)
			cv::rectangle(GreenBox, boundingRect(FinalContour[0]), cv::Scalar(0, 255, 0), 3);

			//			imshow("Shortlisted Image", CounterAndMask[0]);

			vector<float> bgrFeatures = extractBGRFeatures(CounterAndMask[0], srcI);

			vector<double> combinedFeatures(bgrFeatures.begin(), bgrFeatures.end());

			vector<float> histogram = extractBGRFeatures(CounterAndMask[0], srcI);

			int count = 0;
			for (int i = 0; i < combinedFeatures.size(); i++) {
				if (combinedFeatures[i] != 0)
					cout << "here feature: " << combinedFeatures[i] << endl;
				count += 1;

			}
			cvtColor(CounterAndMask[0], temp, COLOR_GRAY2BGR);
			temp = srcI & temp;
			ImageColor(temp, color);

			cout << "count: " << count << endl << "Dominant Color:" << color[0] << endl << endl;

			int masktype = 0;
			if (color[0] == "red") {
				masktype = 0;//red
			}
			else if (color[0] == "orange") {
				masktype = 1;//orange
			}
			else if (color[0] == "yellow") {
				masktype = 2;//yellow
			}
			else {
				masktype = 3;//blue
			}

			vector<float> shapeFeature;

			int NumberOfVertices = extractVertices(Masks[masktype]);
			vector<double> HuResult = extractHuMoments(Masks[masktype]);

			shapeFeature.push_back(static_cast<float>(NumberOfVertices));

			// Loop through the 7 Hu moments and push them one by one into shapeFeature
			for (size_t i = 0; i < HuResult.size(); ++i) {
				shapeFeature.push_back(static_cast<float>(HuResult[i]));
			}

			//comnbine bgr and shape 
			vector<float> combinedBgrShapeFeature;

			// Combine bgrFeature and shapeFeature
			combinedBgrShapeFeature.insert(combinedBgrShapeFeature.end(), bgrFeatures.begin(), bgrFeatures.end());
			combinedBgrShapeFeature.insert(combinedBgrShapeFeature.end(), shapeFeature.begin(), shapeFeature.end());

			// Push the combined feature vector to trainingData
			trainingData.push_back(combinedBgrShapeFeature);

			//save into trainingData

			if (masktype == 1)
				color[0] = "yellow";

			int Number = labelDetermine(NumberOfVertices, HuResult, color[0]);
			cout << "LabelResult:" << Number << endl << endl;

			//store into labels
			labels.push_back(Number);
		}
		//GreenBox = srcI & GreenBox;
	//	imshow("GreenBox Image", GreenBox);
		waitKey();
		destroyAllWindows();
	}



	//==========================================
		// Split data into 80% train, 20% test
	vector<vector<float>> trainData, testData;
	vector<int> trainLabels, testLabels;
	splitData(trainingData, labels, 0.8, trainData, testData, trainLabels, testLabels);


	cout << "training start............................" << endl;
	cout << "Waiting............................" << endl;
	Ptr<ml::SVM> svm;
	Ptr<ml::KNearest> knn;
	Ptr<ml::RTrees> randomForest;

	//training model =======================================================================
	svm = trainSVM(trainData, trainLabels);
	knn = trainKNN(trainData, trainLabels);
	randomForest = trainForest(trainData, trainLabels);
	cout << "training success............................" << endl;

	// Evaluate on the test data================================================================
	vector<int> svmPredictions, knnPredictions, randomPredictions;


	for (int i = 0; i < testData.size(); i++) {
		int predictedLabel = static_cast<int>(svm->predict(testData[i]));

		svmPredictions.push_back(predictedLabel);
	}
	for (int i = 0; i < testData.size(); i++) {
		int predictedLabel = static_cast<int>(knn->predict(testData[i]));

		knnPredictions.push_back(predictedLabel);
	}
	for (int i = 0; i < testData.size(); i++) {
		int predictedLabel = static_cast<int>(randomForest->predict(testData[i]));

		randomPredictions.push_back(predictedLabel);
	}




	// Calculate accuracy for each model
	auto calculateAccuracy = [](const vector<int>& trueLabels, const vector<int>& predictedLabels) -> double {
		int correct = 0;
		for (size_t i = 0; i < trueLabels.size(); ++i) {
			if (trueLabels[i] == predictedLabels[i]) {
				++correct;
			}
		}
		return static_cast<double>(correct) / trueLabels.size() * 100.0;
		};


	if (!svmPredictions.empty()) {
		double accuracySVM = calculateAccuracy(testLabels, svmPredictions);
		cout << "Accuracy of SVM: " << accuracySVM << "%" << endl;
	}

	if (!knnPredictions.empty()) {
		double accuracyKNN = calculateAccuracy(testLabels, knnPredictions);
		cout << "Accuracy of k-NN: " << accuracyKNN << "%" << endl;
	}

	if (!randomPredictions.empty()) {
		double accuracyRF = calculateAccuracy(testLabels, randomPredictions);
		cout << "Accuracy of Random Forest: " << accuracyRF << "%" << endl;
	}
	cout << "end==============" << endl;
	cout << endl;


	// SVM accuracy
	if (!svmPredictions.empty()) {
		double accuracySVM = calculateAccuracy(testLabels, svmPredictions);
		cout << "Accuracy of SVM: " << accuracySVM << "%" << endl;

		double precisionSVM, recallSVM, f1ScoreSVM;
		calculateMetrics(testLabels, svmPredictions, precisionSVM, recallSVM, f1ScoreSVM);
		cout << "SVM Precision: " << precisionSVM << endl;
		cout << "SVM Recall: " << recallSVM << endl;
		cout << "SVM F1-Score: " << f1ScoreSVM << endl;
	}
	cout << endl;

	// k-NN accuracy
	if (!knnPredictions.empty()) {
		double accuracyKNN = calculateAccuracy(testLabels, knnPredictions);
		cout << "Accuracy of k-NN: " << accuracyKNN << "%" << endl;

		double precisionKNN, recallKNN, f1ScoreKNN;
		calculateMetrics(testLabels, knnPredictions, precisionKNN, recallKNN, f1ScoreKNN);
		cout << "k-NN Precision: " << precisionKNN << endl;
		cout << "k-NN Recall: " << recallKNN << endl;
		cout << "k-NN F1-Score: " << f1ScoreKNN << endl;
	}
	cout << endl;

	// Random Forest accuracy
	if (!randomPredictions.empty()) {
		double accuracyRF = calculateAccuracy(testLabels, randomPredictions);
		cout << "Accuracy of Random Forest: " << accuracyRF << "%" << endl;

		double precisionRF, recallRF, f1ScoreRF;
		calculateMetrics(testLabels, randomPredictions, precisionRF, recallRF, f1ScoreRF);
		cout << "Random Forest Precision: " << precisionRF << endl;
		cout << "Random Forest Recall: " << recallRF << endl;
		cout << "Random Forest F1-Score: " << f1ScoreRF << endl;
	}

	return 0;
}