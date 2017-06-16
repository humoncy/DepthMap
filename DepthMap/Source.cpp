#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;
using namespace cv;

const int WINDOW_SIZE = 11; // should be even
const int LEVEL = 6;

void blockMatching(Mat SrcImg1, Mat SrcImg2, Mat& DisparityMap);
void imageShowTesting(Mat& DisparityMap);

int main() {

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg1 = imread("Resources/tsukuba/scene1.row3.col1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg2 = imread("Resources/tsukuba/scene1.row3.col2.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg3 = imread("Resources/tsukuba/scene1.row3.col3.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg4 = imread("Resources/tsukuba/scene1.row3.col4.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg5 = imread("Resources/tsukuba/scene1.row3.col5.ppm", CV_LOAD_IMAGE_COLOR);
	Mat GroundTruth = imread("Resources/tsukuba/truedisp.row3.col3.pgm", CV_LOAD_IMAGE_COLOR);

	cout << "SrcImg1: " << SrcImg1.rows << ' ' << SrcImg1.cols << endl;
	cout << "SrcImg2: " << SrcImg2.rows << ' ' << SrcImg2.cols << endl;

	// Show images
	//imshow("Input Image1", SrcImg1);
	//imshow("Input Image2", SrcImg2);
	//imshow("Input Image3", SrcImg3);
	//imshow("Input Image4", SrcImg4);
	//imshow("Input Image5", SrcImg5);
	imshow("Ground Truth", GroundTruth);

	Mat DisparityMap(SrcImg1.rows, SrcImg1.cols, CV_8UC1, Scalar(0));

	//imageShowTesting(DisparityMap);
	blockMatching(SrcImg1, SrcImg2, DisparityMap);

	imshow("Disparity Map", DisparityMap);
	imwrite("Resources/tsukuba/disparity map.jpg", DisparityMap);
	// Write output images
	//imwrite("Resources/tsukuba/scene1.row3.col1.jpg", SrcImg1);
	//imwrite("Resources/tsukuba/scene1.row3.col2.jpg", SrcImg2);
	/*imwrite("Resources/tsukuba/scene1.row3.col3.jpg", SrcImg3);
	imwrite("Resources/tsukuba/scene1.row3.col4.jpg", SrcImg4);
	imwrite("Resources/tsukuba/scene1.row3.col5.jpg", SrcImg5);
	imwrite("Resources/tsukuba/truedisp.row3.col3.jpg", SrcImg6);*/

	waitKey(0);
	return 0;
}

void blockMatching(Mat SrcImg1, Mat SrcImg2, Mat & DisparityMap)
{
	Mat original_disparitiy(SrcImg1.rows, SrcImg1.cols, CV_32S, Scalar(0));

	int m = ( WINDOW_SIZE - 1 ) / 2;
	for (int rowIndex = m; rowIndex < SrcImg1.rows - m - 1; rowIndex++) {
		for (int colIndex = m; colIndex < SrcImg1.cols - m - 1; colIndex++) {
			int min_ssd = std::numeric_limits<int>::max();

			int search_start = colIndex > 80 ? colIndex - 50 : m;
			// the corresponding epipolar line
			for (int rcolIndex = search_start; rcolIndex < colIndex; rcolIndex++) {
				int d = colIndex - rcolIndex;
				unsigned int ssd = 0;
				for (int i = -m; i <= m; i++) {
					for (int j = -m; j <= m; j++) {
						ssd += pow((int)SrcImg1.at<uchar>(rowIndex + i, colIndex + j) - (int)SrcImg2.at<uchar>(rowIndex + i, colIndex - d + j), 2.0);
					}
				}
				if (ssd < min_ssd) {
					min_ssd = ssd;
					original_disparitiy.at<int>(rowIndex, colIndex) = abs(d);
				}
			}
			//cout <<	original_disparitiy.at<int>(rowIndex, colIndex) << ',';
		}
		//cout << endl;
		//break;
	}

	//Mat normalized_original_disparitiy(SrcImg1.rows, SrcImg1.cols, CV_32FC1, Scalar(0));
	normalize(original_disparitiy, original_disparitiy, 0.0, 255.0, NORM_MINMAX);
	original_disparitiy.convertTo(original_disparitiy, CV_8U, 1.0, 0);
	imshow("TEST", original_disparitiy);
	
	DisparityMap = original_disparitiy;

	// visualization
	/*for (int rowIndex = 0; rowIndex < SrcImg1.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < SrcImg1.cols; colIndex++) {
			int which_level = (float)original_disparitiy.at<uchar>(rowIndex, colIndex) / (255.0 / LEVEL);
			DisparityMap.at<uchar>(rowIndex, colIndex) = ((float)which_level / LEVEL) * 255.0;
		}
	}*/
}

void imageShowTesting(Mat & DisparityMap)
{
	/*Mat mat1(2, 2, CV_32FC1);
	mat1.at<float>(0, 0) = 1.0f;
	mat1.at<float>(0, 1) = 2.0f;
	mat1.at<float>(1, 0) = 3.0f;
	mat1.at<float>(1, 1) = 4.0f;
	cout << "Mat 1:\n" << mat1 << endl;
	normalize(mat1, mat1, 0.0, 255.0, NORM_MINMAX);
	cout << "Normalized Mat 1:\n" << mat1 << endl;*/
	//imshow("Test", mat1);

	/*float temp = (float)1 / LEVEL;
	int x = temp * 255.0;
	cout << x << endl;*/
	//cout << abs(-255.0);

	Mat original_disparitiy(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));

	for (int rowIndex = 0; rowIndex < DisparityMap.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < DisparityMap.cols; colIndex++) {
			DisparityMap.at<uchar>(rowIndex, colIndex) = (float)colIndex / DisparityMap.cols * 255;
			original_disparitiy.at<int>(rowIndex, colIndex) = (float)colIndex / DisparityMap.cols * 255;
		}
	}

	normalize(original_disparitiy, original_disparitiy, 0.0, 255.0, NORM_MINMAX);
	original_disparitiy.convertTo(original_disparitiy, CV_8U, 1.0, 0);
	imshow("original disparity", original_disparitiy);

	// visualization
	for (int rowIndex = 0; rowIndex < DisparityMap.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < DisparityMap.cols; colIndex++) {
			int which_level = (float)DisparityMap.at<uchar>(rowIndex, colIndex) / (255.0 / LEVEL);
			DisparityMap.at<uchar>(rowIndex, colIndex) = ((float)which_level / LEVEL) * 255.0;
		}
	}
}
