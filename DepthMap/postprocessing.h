#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

// utility
float filterblurpoint(cv::Mat newimage, int i, int j, int filtersize);
float filterblurpointRGB(cv::Mat newimage, int i, int j, int filtersize, int color);
float calculateGradient(float dx, float dy, cv::Mat DisparityMap, float d1);

void disparityRefinement(cv::Mat & disparityMap, cv::Mat & refinement_disparitymap, const int threshold);
void depthOfField(cv::Mat & SrcImg, cv::Mat & DstImg, cv::Mat & disparityMap);
void depthOfFieldColor(cv::Mat & SrcImg, cv::Mat & DstImg, cv::Mat & disparityMap);
void droppingleaves(cv::Mat & SrcImg, string & AnimationDir, vector<string> & files, cv::Mat & disparityMap);

#endif // POSTPROCESSING_H