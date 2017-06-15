#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main() {

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg1 = imread("Resources/tsukuba/scene1.row3.col1.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg2 = imread("Resources/tsukuba/scene1.row3.col2.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg3 = imread("Resources/tsukuba/scene1.row3.col3.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg4 = imread("Resources/tsukuba/scene1.row3.col4.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg5 = imread("Resources/tsukuba/scene1.row3.col5.ppm", CV_LOAD_IMAGE_COLOR);
	Mat SrcImg6 = imread("Resources/tsukuba/truedisp.row3.col3.pgm", CV_LOAD_IMAGE_COLOR);

	// Show images
	imshow("Input Image1", SrcImg1);
	imshow("Input Image2", SrcImg2);
	imshow("Input Image3", SrcImg3);
	imshow("Input Image4", SrcImg4);
	imshow("Input Image5", SrcImg5);
	imshow("Input Image6", SrcImg6);

	// Write output images
	imwrite("Resources/tsukuba/scene1.row3.col1.jpg", SrcImg1);
	imwrite("Resources/tsukuba/scene1.row3.col2.jpg", SrcImg2);
	imwrite("Resources/tsukuba/scene1.row3.col3.jpg", SrcImg3);
	imwrite("Resources/tsukuba/scene1.row3.col4.jpg", SrcImg4);
	imwrite("Resources/tsukuba/scene1.row3.col5.jpg", SrcImg5);
	imwrite("Resources/tsukuba/truedisp.row3.col3.jpg", SrcImg6);

	waitKey(0);
	return 0;
}