#ifndef STEREOMATCHING_H
#define STEREOMATCHING_H

#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

#define DISPARITY_RANGE 20
#define INT_MAX std::numeric_limits<int>::max();

void blockMatching(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat& DisparityMap);
void fasterblockMatching(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat& DisparityMap);
void matchingDP(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat& DisparityMap);

#endif // STEREOMATCHING_H