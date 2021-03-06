#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <limits>
#include <Windows.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "gaussian.h"
#include "postprocessing.h"
#include "stereomatching.h"

#define BLUR_RADIUS 3
#define PATHS_PER_SCAN 8
#define MAX_SHORT numeric_limits<unsigned short>::max()
#define SMALL_PENALTY 3
#define LARGE_PENALTY 30
#define DEBUG false
#define CV_EVENT_LBUTTONDOWN 1 
#define CV_EVENT_RBUTTONDOWN 2     
#define LEVEL 30

const int WINDOW_SIZE = 7; // should be odd

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




void depthOfFieldColorchange(cv::Mat & SrcImg, cv::Mat & DstImg, cv::Mat & disparityMap, int disparity)
{
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			//if ((float)disparityMap.at<uchar>(i, j) < disparity-8 || (float)disparityMap.at<uchar>(i, j) > disparity + 8)
			if ((float)disparityMap.at<uchar>(i, j) != disparity)
			{
				DstImg.at<Vec3b>(i, j)[0] = filterblurpointRGB(SrcImg, i, j, 5, 0);
				DstImg.at<Vec3b>(i, j)[1] = filterblurpointRGB(SrcImg, i, j, 5, 1);
				DstImg.at<Vec3b>(i, j)[2] = filterblurpointRGB(SrcImg, i, j, 5, 2);
			}
		}
	}
}

CvPoint VertexOne;
Mat color_img_mouse = imread("Resources/tsukuba/scene1.row3.col1.jpg", CV_LOAD_IMAGE_COLOR);
Mat disparity_mouse = Mat(Size(color_img_mouse.cols, color_img_mouse.rows), CV_8UC1, Scalar::all(0));
void onMouse(int event, int x, int y, int flag, void* param) {
	Mat color_img_mouse_tmp = color_img_mouse.clone();
	if (event == 1) {//按左鍵
		VertexOne = cvPoint(x, y);
		int disparity = disparity_mouse.at<uchar>(y, x);//剛好row col與x y相反
		cout << x << " " << y << " " << (float)disparity_mouse.at<uchar>(y, x) << endl;
		depthOfFieldColorchange(color_img_mouse, color_img_mouse_tmp, disparity_mouse, disparity);
		imshow("Depth of Field (Mouse)", color_img_mouse_tmp);
	}
}

int getdir(string dir, vector<string> &files) {
	DIR *dp;//創立資料夾指標
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}
	while ((dirp = readdir(dp)) != NULL) {//如果dirent指標非空
		files.push_back(string(dirp->d_name));//將資料夾和檔案名放入vector
	}
	closedir(dp);//關閉資料夾指標
	return 0;
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

	printf("SGM done in %.2lf seconds.\n", elapsed_secs);

	char * outFileName = "Resources/tsukuba/disparity map sgm.jpg";

	//Refinement the steroe image
	//Mat Refinement = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));
	Mat refinement_disparitymap = disparityMap.clone();
	const int threshold = 1.8;//disparity gradient limit
	disparityRefinement(disparityMap, refinement_disparitymap, threshold);

	saveDisparityMap(disparityMap, disparityRange, outFileName);
	imshow("Disparity Map SGM", disparityMap);
	saveDisparityMap(refinement_disparitymap, disparityRange, "Resources/tsukuba/refinement map sgm.jpg");
	//imshow("refinement", refinement_disparitymap);
	medianBlur(refinement_disparitymap, refinement_disparitymap, 3);
	imshow("Disparity Map SGM after Refinement", refinement_disparitymap);

	// checking refinements
	/*Mat check = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			check.at<uchar>(i, j) = abs((float)refinement_disparitymap.at<uchar>(i, j) - (float)disparityMap.at<uchar>(i, j));
		}
	}
	imshow("check", check);*/

	// depth of field for gray image
	//Mat dof_gray = firstImage.clone();
	//depthOfField(dof_gray, dof_gray, refinement_disparitymap);
	//imshow("Depth of Field (gray)", dof_gray);

	//Mat DisparityMap_BM(firstImage.rows, firstImage.cols, CV_8UC1, Scalar(0));
	Mat DisparityMap2_FBM(firstImage.rows, firstImage.cols, CV_8UC1, Scalar(0));
	Mat DisparityMap3_DP(firstImage.rows, firstImage.cols, CV_8UC1, Scalar(0));

	begin = clock();

	//blockMatching(WINDOW_SIZE, firstImage, secondImage, DisparityMap_BM);
	fasterblockMatching(WINDOW_SIZE, firstImage, secondImage, DisparityMap2_FBM);
	matchingDP(WINDOW_SIZE, firstImage, secondImage, DisparityMap3_DP);
	medianBlur(DisparityMap2_FBM, DisparityMap2_FBM, 5);
	medianBlur(DisparityMap3_DP, DisparityMap3_DP, 5);

	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("DP done in %.2lf seconds.\n", elapsed_secs);

	imshow("Disparity Map Block Matching", DisparityMap2_FBM);
	imshow("Disparity Map Dynamic Programming", DisparityMap3_DP);

	Mat color_img = imread("Resources/tsukuba/scene1.row3.col1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat dofColor = color_img.clone();
	depthOfFieldColor(color_img, dofColor, refinement_disparitymap);
	imshow("Depth of Field", dofColor);


	disparity_mouse = refinement_disparitymap.clone();
	cvNamedWindow("Depth of Field (Mouse)", 1);
	cvResizeWindow("Depth of Field (Mouse)", color_img.cols, color_img.rows);
	//for mouse event
	cvSetMouseCallback("Depth of Field (Mouse)", onMouse, NULL);//設定滑鼠callback函式，改變disparity map 的景深影像
	imshow("Depth of Field (Mouse)", color_img_mouse);
	for (int rowIndex = 0; rowIndex < firstImage.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < secondImage.cols; colIndex++) {
			int which_level = (float)disparity_mouse.at<uchar>(rowIndex, colIndex) / (255.0 / LEVEL);
			disparity_mouse.at<uchar>(rowIndex, colIndex) = ((float)which_level / LEVEL) * 255.0;
		}
	}
	imshow("disparity_mouse", disparity_mouse);
	//for mouse event
	int key = 0;
	while (true) {
		key = cvWaitKey(0);
		if (key == 27) {
			break;
		}
	}

	string animationDir = "./Resources/maple_leaves/";
	vector<string> files = vector<string>();
	getdir(animationDir, files);
	droppingleaves(color_img, animationDir, files, refinement_disparitymap);

	//waitKey(0);
	return 0;
}
