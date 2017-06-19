#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "gaussian.h"

#define BLUR_RADIUS 3
#define PATHS_PER_SCAN 8
#define MAX_SHORT numeric_limits<unsigned short>::max()
#define SMALL_PENALTY 3
#define LARGE_PENALTY 30
#define DEBUG false

using namespace std;
using namespace cv;

struct path {
	short rowDiff;
	short colDiff;
	short index;
};

inline uchar getMatElement(Mat &matrix, int i, int j) {
	assert(matrix.type() == CV_8UC1);
	if (i < 0 || i >= matrix.rows || j < 0 || j >= matrix.cols) {
		return 0;
	}
	else {
		return matrix.at<uchar>(i, j);
	}
}

// for debug
void printArray(unsigned short ***array, int rows, int cols, int depth) {
	for (int d = 0; d < depth; ++d) {
		cout << "disparity: " << d << endl;
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				cout << "\t" << array[row][col][d];
			}
			cout << endl;
		}
	}
}

unsigned short calculatePixelCostOneWayBT(int row, int leftCol, int rightCol, const Mat &leftImage, const Mat &rightImage) {

	char leftValue, rightValue, beforeRightValue, afterRightValue, rightValueMinus, rightValuePlus, rightValueMin, rightValueMax;

	if (leftCol<0)
		leftValue = 0;
	else
		leftValue = leftImage.at<uchar>(row, leftCol);

	if (rightCol<0)
		rightValue = 0;
	else
		rightValue = rightImage.at<uchar>(row, rightCol);

	if (rightCol > 0) {
		beforeRightValue = rightImage.at<uchar>(row, rightCol - 1);
	}
	else {
		beforeRightValue = rightValue;
	}

	//cout << rightCol <<" " <<leftCol<< endl;   	
	if (rightCol + 1 < rightImage.cols && rightCol>0) {
		afterRightValue = rightImage.at<uchar>(row, rightCol + 1);
	}
	else {
		afterRightValue = rightValue;
	}

	rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
	rightValuePlus = round((rightValue + afterRightValue) / 2.f);

	rightValueMin = min(rightValue, min(rightValueMinus, rightValuePlus));
	rightValueMax = max(rightValue, max(rightValueMinus, rightValuePlus));

	return max(0, max((leftValue - rightValueMax), (rightValueMin - leftValue)));
}

inline unsigned short calculatePixelCostBT(int row, int leftCol, int rightCol, const Mat &leftImage, const Mat &rightImage) {
	return min(calculatePixelCostOneWayBT(row, leftCol, rightCol, leftImage, rightImage),
		calculatePixelCostOneWayBT(row, rightCol, leftCol, rightImage, leftImage));
}

void calculatePixelCost(Mat &firstImage, Mat &secondImage, int disparityRange, unsigned short ***C)
{
	for (int row = 0; row < firstImage.rows; ++row) {
		for (int col = 0; col < firstImage.cols; ++col) {
			for (int d = 0; d < disparityRange; ++d) {
				//cout << col << " " << d << " "<<disparityRange << " " << col - d << endl;
				C[row][col][d] = calculatePixelCostBT(row, col, col - d, firstImage, secondImage);
			}
		}
	}
}

// pathCount can be 1, 2, 4, or 8
void initializeFirstScanPaths(vector<path> &paths, unsigned short pathCount) {
	for (unsigned short i = 0; i < pathCount; ++i) {
		paths.push_back(path());
	}

	if (paths.size() >= 1) {
		paths[0].rowDiff = 0;
		paths[0].colDiff = -1;
		paths[0].index = 1;
	}

	if (paths.size() >= 2) {
		paths[1].rowDiff = -1;
		paths[1].colDiff = 0;
		paths[1].index = 2;
	}

	if (paths.size() >= 4) {
		paths[2].rowDiff = -1;
		paths[2].colDiff = 1;
		paths[2].index = 4;

		paths[3].rowDiff = -1;
		paths[3].colDiff = -1;
		paths[3].index = 7;
	}

	if (paths.size() >= 8) {
		paths[4].rowDiff = -2;
		paths[4].colDiff = 1;
		paths[4].index = 8;

		paths[5].rowDiff = -2;
		paths[5].colDiff = -1;
		paths[5].index = 9;

		paths[6].rowDiff = -1;
		paths[6].colDiff = -2;
		paths[6].index = 13;

		paths[7].rowDiff = -1;
		paths[7].colDiff = 2;
		paths[7].index = 15;
	}
}

// pathCount can be 1, 2, 4, or 8
void initializeSecondScanPaths(vector<path> &paths, unsigned short pathCount) {
	for (unsigned short i = 0; i < pathCount; ++i) {
		paths.push_back(path());
	}

	if (paths.size() >= 1) {
		paths[0].rowDiff = 0;
		paths[0].colDiff = 1;
		paths[0].index = 0;
	}

	if (paths.size() >= 2) {
		paths[1].rowDiff = 1;
		paths[1].colDiff = 0;
		paths[1].index = 3;
	}

	if (paths.size() >= 4) {
		paths[2].rowDiff = 1;
		paths[2].colDiff = 1;
		paths[2].index = 5;

		paths[3].rowDiff = 1;
		paths[3].colDiff = -1;
		paths[3].index = 6;
	}

	if (paths.size() >= 8) {
		paths[4].rowDiff = 2;
		paths[4].colDiff = 1;
		paths[4].index = 10;

		paths[5].rowDiff = 2;
		paths[5].colDiff = -1;
		paths[5].index = 11;

		paths[6].rowDiff = 1;
		paths[6].colDiff = -2;
		paths[6].index = 12;

		paths[7].rowDiff = 1;
		paths[7].colDiff = 2;
		paths[7].index = 14;
	}
}

unsigned short aggregateCost(int row, int col, int d, path &p, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***A) {
	unsigned short aggregatedCost = 0;
	aggregatedCost += C[row][col][d];

	if (DEBUG) {
		printf("{P%d}[%d][%d](d%d)\n", p.index, row, col, d);
	}

	if (row + p.rowDiff < 0 || row + p.rowDiff >= rows || col + p.colDiff < 0 || col + p.colDiff >= cols) {
		// border
		A[row][col][d] += aggregatedCost;
		if (DEBUG) {
			printf("{P%d}[%d][%d](d%d)-> %d <BORDER>\n", p.index, row, col, d, A[row][col][d]);
		}
		return A[row][col][d];
	}

	unsigned short minPrev, minPrevOther, prev, prevPlus, prevMinus;
	prev = minPrev = minPrevOther = prevPlus = prevMinus = MAX_SHORT;

	for (int disp = 0; disp < disparityRange; ++disp) {
		unsigned short tmp = A[row + p.rowDiff][col + p.colDiff][disp];
		if (minPrev > tmp) {
			minPrev = tmp;
		}

		if (disp == d) {
			prev = tmp;
		}
		else if (disp == d + 1) {
			prevPlus = tmp;
		}
		else if (disp == d - 1) {
			prevMinus = tmp;
		}
		else {
			if (minPrevOther > tmp + LARGE_PENALTY) {
				minPrevOther = tmp + LARGE_PENALTY;
			}
		}
	}

	aggregatedCost += min(min((int)prevPlus + SMALL_PENALTY, (int)prevMinus + SMALL_PENALTY), min((int)prev, (int)minPrevOther + LARGE_PENALTY));
	aggregatedCost -= minPrev;

	A[row][col][d] += aggregatedCost;

	if (DEBUG) {
		printf("{P%d}[%d][%d](d%d)-> %d<CALCULATED>\n", p.index, row, col, d, A[row][col][d]);
	}

	return A[row][col][d];
}

float printProgress(unsigned int current, unsigned int max, int lastProgressPrinted) {
	int progress = floor(100 * current / (float)max);
	if (progress >= lastProgressPrinted + 20) {
		lastProgressPrinted = lastProgressPrinted + 20;
		cout << lastProgressPrinted << "%" << endl;
	}
	return lastProgressPrinted;
}

void aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S) {
	vector<path> firstScanPaths;
	vector<path> secondScanPaths;

	initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
	initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

	int lastProgressPrinted = 0;
	cout << "First scan..." << endl;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			for (unsigned int path = 0; path < firstScanPaths.size(); ++path) {
				for (int d = 0; d < disparityRange; ++d) {
					S[row][col][d] += aggregateCost(row, col, d, firstScanPaths[path], rows, cols, disparityRange, C, A[path]);
				}
			}
		}
		lastProgressPrinted = printProgress(row, rows - 1, lastProgressPrinted);
	}

	lastProgressPrinted = 0;
	cout << "Second scan..." << endl;
	for (int row = rows - 1; row >= 0; --row) {
		for (int col = cols - 1; col >= 0; --col) {
			for (unsigned int path = 0; path < secondScanPaths.size(); ++path) {
				for (int d = 0; d < disparityRange; ++d) {
					S[row][col][d] += aggregateCost(row, col, d, secondScanPaths[path], rows, cols, disparityRange, C, A[path]);
				}
			}
		}
		lastProgressPrinted = printProgress(rows - 1 - row, rows - 1, lastProgressPrinted);
	}
}

void computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, Mat &disparityMap) {
	unsigned int disparity = 0, minCost;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			minCost = MAX_SHORT;
			for (int d = disparityRange - 1; d >= 0; --d) {
				if (minCost > S[row][col][d]) {
					minCost = S[row][col][d];
					disparity = d;
				}
			}
			disparityMap.at<uchar>(row, col) = disparity;
		}
	}
}

void saveDisparityMap(Mat &disparityMap, int disparityRange, char* outputFile) {
	double factor = 256.0 / disparityRange;
	for (int row = 0; row < disparityMap.rows; ++row) {
		for (int col = 0; col < disparityMap.cols; ++col) {
			disparityMap.at<uchar>(row, col) *= factor;
		}
	}
	imwrite(outputFile, disparityMap);
}
float mean(Mat disparityMap)
{
	float tmp = 0.0;
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			tmp += disparityMap.at<uchar>(i, j);
		}
	}
	tmp /= (disparityMap.rows*disparityMap.cols);
	return tmp;
}
//∫‚DG
float calculateGradient(float dx, float dy, Mat DisparityMap, float d1)
{

	float d2 = getMatElement(DisparityMap, dx, dy);
	float DG = ((abs(d2 - d1)) / abs((1 + (d2 - d1) / 2)));
	return DG;
}
float filterblurpoint(Mat newimage, int i, int j, int filtersize) {
	int filter = (filtersize - 1) / 2;
	float tmp = 0.0;
	for (int a = -filter; a <= filter; a++) {
		for (int b = -filter; b <= filter; b++) {
			tmp += (float)getMatElement(newimage, i + a, j + b);
		}
	}
	return tmp / (filtersize*filtersize);
}

float filterblurpointRGB(Mat newimage, int i, int j, int filtersize, int color) {
	int filter = (filtersize - 1) / 2;
	float tmp = 0.0;
	for (int a = -filter; a <= filter; a++) {
		for (int b = -filter; b <= filter; b++) {
			tmp += (float)getMatElement(newimage, i + a, j + b);
		}
	}
	return tmp / (filtersize*filtersize);
}

int main() {

	Mat firstImage = imread("Resources/tsukuba/scene1.row3.col1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat secondImage = imread("Resources/tsukuba/scene1.row3.col2.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	if (!firstImage.data || !secondImage.data) {
		cerr << "Could not open or find one of the images!" << endl;
		return -1;
	}

	unsigned int disparityRange = 25;
	unsigned short ***C; // pixel cost array W x H x D
	unsigned short ***S; // aggregated cost array W x H x D
	unsigned short ****A; // single path cost array PATHS_PER_SCAN x W x H x D
						  /*
						  * TODO
						  * variable LARGE_PENALTY
						  */

	clock_t begin = clock();

	cout << "Allocating space..." << endl;

	// allocate cost arrays
	C = new unsigned short**[firstImage.rows];
	S = new unsigned short**[firstImage.rows];
	for (int row = 0; row < firstImage.rows; ++row) {
		C[row] = new unsigned short*[firstImage.cols];
		S[row] = new unsigned short*[firstImage.cols]();
		for (int col = 0; col < firstImage.cols; ++col) {
			C[row][col] = new unsigned short[disparityRange];
			S[row][col] = new unsigned short[disparityRange](); // initialize to 0
		}
	}

	A = new unsigned short ***[PATHS_PER_SCAN];
	for (int path = 0; path < PATHS_PER_SCAN; ++path) {
		A[path] = new unsigned short **[firstImage.rows];
		for (int row = 0; row < firstImage.rows; ++row) {
			A[path][row] = new unsigned short*[firstImage.cols];
			for (int col = 0; col < firstImage.cols; ++col) {
				A[path][row][col] = new unsigned short[disparityRange];
				for (unsigned int d = 0; d < disparityRange; ++d) {
					A[path][row][col][d] = 0;
				}
			}
		}
	}

	cout << "Smoothing images..." << endl;
	// 
	grayscaleGaussianBlur(firstImage, firstImage, BLUR_RADIUS);
	grayscaleGaussianBlur(secondImage, secondImage, BLUR_RADIUS);

	cout << "Calculating pixel cost for the image..." << endl;
	calculatePixelCost(firstImage, secondImage, disparityRange, C);
	if (DEBUG) {
		printArray(C, firstImage.rows, firstImage.cols, disparityRange);
	}
	cout << "Aggregating costs..." << endl;
	aggregateCosts(firstImage.rows, firstImage.cols, disparityRange, C, A, S);

	Mat disparityMap = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));

	cout << "Computing disparity..." << endl;
	computeDisparity(S, firstImage.rows, firstImage.cols, disparityRange, disparityMap);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Done in %.2lf seconds.\n", elapsed_secs);

	char * outFileName = "Resources/tsukuba/disparity map sgm.jpg";
	//Refinement the steroe image
	//Mat Refinement = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));
	Mat Refinement = disparityMap.clone();
	float disimg_mean = mean(disparityMap);
	int threshold = 1.8;//disparity gradient limit
	for (int i = 1; i < disparityMap.rows - 1; ++i) {
		for (int j = 1; j < disparityMap.cols - 1; ++j) {
			float d1 = disparityMap.at<uchar>(i, j);//the current disparity
			float up, down, right, left;//save the gradient
			left = calculateGradient(i - 1, j, disparityMap, d1);
			right = calculateGradient(i + 1, j, disparityMap, d1);
			up = calculateGradient(i, j - 1, disparityMap, d1);
			down = calculateGradient(i, j + 1, disparityMap, d1);
			if (left > threshold)//check the gradient
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i - 1, j) - disimg_mean))
				{
					Refinement.at<uchar>(i - 1, j) = d1;
				}
				else d1 = disparityMap.at<uchar>(i - 1, j);
			}
			if (up > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i, j - 1) - disimg_mean))
				{
					Refinement.at<uchar>(i, j - 1) = d1;
				}
				else d1 = disparityMap.at<uchar>(i, j - 1);
			}
			if (right > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i + 1, j) - disimg_mean))
				{
					Refinement.at<uchar>(i + 1, j) = d1;
				}
				else d1 = disparityMap.at<uchar>(i + 1, j);
			}
			if (down > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i, j + 1) - disimg_mean))
				{
					Refinement.at<uchar>(i, j + 1) = d1;
				}
				else d1 = (float)disparityMap.at<uchar>(i, j + 1);
			}
			//Refinement.at<uchar>(i, j) = d1;
		}
	}

	saveDisparityMap(Refinement, disparityRange, "Resources/tsukuba/refinement map sgm.jpg");
	imshow("refinement", Refinement);
	medianBlur(Refinement, Refinement, 3);
	imshow("refinement_median", Refinement);

	saveDisparityMap(disparityMap, disparityRange, outFileName);
	imshow("Disparity Map", disparityMap);

	Mat check = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			check.at<uchar>(i, j) = abs((float)Refinement.at<uchar>(i, j) - (float)disparityMap.at<uchar>(i, j));
		}
	}
	imshow("check", check);

	//¥∫≤`for gray image
	Mat newimage = firstImage.clone();
	//Mat checknew = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));
	//Mat checknew = firstImage.clone();
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			if ((float)Refinement.at<uchar>(i, j) <= 80)
			{
				//checknew.at<uchar>(i, j) = newimage.at<uchar>(i, j);
				newimage.at<uchar>(i, j) = filterblurpoint(newimage, i, j, 5);
			}
			if ((float)Refinement.at<uchar>(i, j) > 80 && (float)Refinement.at<uchar>(i, j) <= 100)
			{
				//checknew.at<uchar>(i, j) = newimage.at<uchar>(i, j);
				newimage.at<uchar>(i, j) = filterblurpoint(newimage, i, j, 3);
			}
			//if ((float)Refinement.at<uchar>(i, j) <= 100)
			//{
			//	checknew.at<uchar>(i, j) = filterblurpoint(newimage, i, j, 5);
			//}
		}
		cout << endl;
	}
	imshow("sense", newimage);

	//Mat color_img = imread("Resources/tsukuba/scene1.row3.col1.jpg", CV_LOAD_IMAGE_COLOR);
	//for (int i = 0; i < disparityMap.rows; i++) {
	//	for (int j = 0; j < disparityMap.cols; j++) {
	//		if ((float)Refinement.at<uchar>(i, j) <= 80)
	//		{
	//			color_img.at<Vec3b>(i, j)[0] = filterblurpoint(color_img, i, j, 5);
	//			color_img.at<Vec3b>(i, j)[1] = filterblurpoint(color_img, i, j, 5);
	//			color_img.at<Vec3b>(i, j)[2] = filterblurpoint(color_img, i, j, 5);

	//		}
	//		if ((float)Refinement.at<uchar>(i, j) > 80 && (float)Refinement.at<uchar>(i, j) <= 100)
	//		{
	//			color_img.at<Vec3b>(i, j)[0] = filterblurpoint(color_img, i, j, 3);
	//			color_img.at<Vec3b>(i, j)[1] = filterblurpoint(color_img, i, j, 3);
	//			color_img.at<Vec3b>(i, j)[2] = filterblurpoint(color_img, i, j, 3);
	//		}
	//	}
	//	cout << endl;
	//}
	//imshow("color_img", color_img);
	//imshow("checknew", checknew);
	//for (int i = 0; i < disparityMap.rows; i++) {
	//	for (int j = 0; j < disparityMap.cols; j++) {
	//		if ((float)disparityMap.at<uchar>(i, j) == 0)
	//		{
	//			cout << (float)disparityMap.at<uchar>(i, j) << " ";
	//		}
	//	}
	//	cout << endl;
	//}

	waitKey(0);
	return 0;
}
