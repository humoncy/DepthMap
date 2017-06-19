#include "postprocessing.h"

// simple wrapper for accessing single-channel matrix elements
inline uchar getMatElement(cv::Mat &matrix, int i, int j) {
	assert(matrix.type() == CV_8UC1);
	if (i < 0 || i >= matrix.rows || j < 0 || j >= matrix.cols) {
		return 0;
	}
	else {
		return matrix.at<uchar>(i, j);
	}
}

inline uchar getMatElementRGB(Mat &matrix, int i, int j, int color) {
	assert(matrix.type() == CV_8UC3);
	if (i < 0 || i >= matrix.rows || j < 0 || j >= matrix.cols) {
		return 0;
	}
	else
	{
		return matrix.at<Vec3b>(i, j)[color];
	}
}

float filterblurpoint(cv::Mat newimage, int i, int j, int filtersize) {
	int filter = (filtersize - 1) / 2;
	float tmp = 0.0;
	for (int a = -filter; a <= filter; a++) {
		for (int b = -filter; b <= filter; b++) {
			tmp += (float)getMatElement(newimage, i + a, j + b);
		}
	}
	return tmp / (filtersize*filtersize);
}

float filterblurpointRGB(cv::Mat newimage, int i, int j, int filtersize, int color) {
	int filter = (filtersize - 1) / 2;
	float tmp = 0.0;
	for (int a = -filter; a <= filter; a++) {
		for (int b = -filter; b <= filter; b++) {
			tmp += (float)getMatElementRGB(newimage, i + a, j + b, color);
		}
	}
	return tmp / (filtersize*filtersize);
}

float calculateGradient(float dx, float dy, cv::Mat DisparityMap, float d1)
{

	float d2 = getMatElement(DisparityMap, dx, dy);
	float DG = ((abs(d2 - d1)) / abs((1 + (d2 - d1) / 2)));
	return DG;
}

void disparityRefinement(cv::Mat & disparityMap, cv::Mat & refinement_disparitymap, const int threshold)
{
	float disimg_mean = mean(disparityMap)[0];
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
					refinement_disparitymap.at<uchar>(i - 1, j) = d1;
				}
				else d1 = disparityMap.at<uchar>(i - 1, j);
			}
			if (up > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i, j - 1) - disimg_mean))
				{
					refinement_disparitymap.at<uchar>(i, j - 1) = d1;
				}
				else d1 = disparityMap.at<uchar>(i, j - 1);
			}
			if (right > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i + 1, j) - disimg_mean))
				{
					refinement_disparitymap.at<uchar>(i + 1, j) = d1;
				}
				else d1 = disparityMap.at<uchar>(i + 1, j);
			}
			if (down > threshold)
			{
				if (abs(d1 - disimg_mean) < abs((float)disparityMap.at<uchar>(i, j + 1) - disimg_mean))
				{
					refinement_disparitymap.at<uchar>(i, j + 1) = d1;
				}
				else d1 = (float)disparityMap.at<uchar>(i, j + 1);
			}
			//Refinement.at<uchar>(i, j) = d1;
		}
	}
}

void depthOfField(cv::Mat & SrcImg, cv::Mat & DstImg, cv::Mat & disparityMap)
{
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			if ((float)disparityMap.at<uchar>(i, j) <= 80)
			{
				//checknew.at<uchar>(i, j) = newimage.at<uchar>(i, j);
				DstImg.at<uchar>(i, j) = filterblurpoint(SrcImg, i, j, 5);
			}
			if ((float)disparityMap.at<uchar>(i, j) > 80 && (float)disparityMap.at<uchar>(i, j) <= 100)
			{
				//checknew.at<uchar>(i, j) = newimage.at<uchar>(i, j);
				DstImg.at<uchar>(i, j) = filterblurpoint(SrcImg, i, j, 3);
			}
		}
	}
}

void depthOfFieldColor(cv::Mat & SrcImg, cv::Mat & DstImg, cv::Mat & disparityMap)
{
	for (int i = 0; i < disparityMap.rows; i++) {
		for (int j = 0; j < disparityMap.cols; j++) {
			if ((float)disparityMap.at<uchar>(i, j) <= 80)
			{
				DstImg.at<Vec3b>(i, j)[0] = filterblurpointRGB(SrcImg, i, j, 5, 0);
				DstImg.at<Vec3b>(i, j)[1] = filterblurpointRGB(SrcImg, i, j, 5, 1);
				DstImg.at<Vec3b>(i, j)[2] = filterblurpointRGB(SrcImg, i, j, 5, 2);

			}
			if ((float)disparityMap.at<uchar>(i, j) > 80 && (float)disparityMap.at<uchar>(i, j) <= 100)
			{
				DstImg.at<Vec3b>(i, j)[0] = filterblurpointRGB(SrcImg, i, j, 3, 0);
				DstImg.at<Vec3b>(i, j)[1] = filterblurpointRGB(SrcImg, i, j, 3, 1);
				DstImg.at<Vec3b>(i, j)[2] = filterblurpointRGB(SrcImg, i, j, 3, 2);
			}
		}
	}
}

bool isWhite(Vec3b color)
{
	if (color[0] > 100 && color[1] > 100 && color[2] > 100) {
		return true;
	}
	return false;
}

void droppingleaves(cv::Mat & SrcImg, string & AnimationDir, vector<string> & files, cv::Mat & disparityMap)
{
	int img_cnt = 2;
	while (1) {

		Mat AniImg = imread(AnimationDir + files[img_cnt]);
		Mat animated_srcimg = SrcImg.clone();

		for (int rowIndex = 0; rowIndex < AniImg.rows; rowIndex++) {
			for (int colIndex = 0; colIndex < animated_srcimg.cols; colIndex++) {
				if ( !isWhite(AniImg.at<Vec3b>(rowIndex, colIndex)) && (int)disparityMap.at<uchar>(rowIndex, colIndex) < 80 ) {
					animated_srcimg.at<Vec3b>(rowIndex, colIndex) = AniImg.at<Vec3b>(rowIndex, colIndex);
				}
				else {
					animated_srcimg.at<Vec3b>(rowIndex, colIndex) = SrcImg.at<Vec3b>(rowIndex, colIndex);
				}
			}
		}
		imshow("Dropping maples", animated_srcimg);
		waitKey(100);
		if (img_cnt < files.size() - 1) {
			img_cnt++;
		}
		else {
			img_cnt = 2;
		}
	}
}
