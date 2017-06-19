#include "stereomatching.h"

/* A utility function that returns minimum of 3 integers */
inline int min_(int x, int y, int z)
{
	if (x < y)
		return (x < z) ? x : z;
	else
		return (y < z) ? y : z;
}

// simple wrapper for accessing single-channel matrix elements
inline int getMatElement(cv::Mat &matrix, int i, int j) {
	assert(matrix.type() == CV_32S);
	if (i < 0 || i >= matrix.rows || j < 0 || j >= matrix.cols) {
		return 0;
	}
	else {
		return matrix.at<int>(i, j);
	}
}

void blockMatching(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat & DisparityMap)
{
	Mat original_disparitiy(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));

	int m = (windowsize - 1) / 2;
	for (int rowIndex = m; rowIndex < DisparityMap.rows - m - 1; rowIndex++) {
		for (int colIndex = m; colIndex < DisparityMap.cols - m - 1; colIndex++) {
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
	original_disparitiy.convertTo(DisparityMap, CV_8U, 1.0, 0);
	imshow("TEST", DisparityMap);
}

void fasterblockMatching(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat & DisparityMap)
{
	Mat original_disparitiy(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));
	Mat min_ssd(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(std::numeric_limits<int>::max()));

	int m = (windowsize - 1) / 2;

	for (int d = 0; d < DISPARITY_RANGE; d++) {
		Mat integral_ssd(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));

		for (int rowIndex = 0; rowIndex < DisparityMap.rows; rowIndex++) {
			int sum = 0;
			for (int colIndex = d; colIndex < DisparityMap.cols; colIndex++) {
				sum += pow((int)SrcImg1.at<uchar>(rowIndex, colIndex) - (int)SrcImg2.at<uchar>(rowIndex, colIndex - d), 2.0);
				if (rowIndex == 0) {
					integral_ssd.at<int>(rowIndex, colIndex) = sum;
				}
				else {
					integral_ssd.at<int>(rowIndex, colIndex) = integral_ssd.at<int>(rowIndex - 1, colIndex) + sum;
				}
			}
		}

		// ignore some margin
		for (int rowIndex = m + 1; rowIndex < DisparityMap.rows - m - 1; rowIndex++) {
			for (int colIndex = m + 1; colIndex < DisparityMap.cols - m - 1; colIndex++) {
				//int ssd = integral_ssd.at<int>(rowIndex + m, colIndex + m) - integral_ssd.at<int>(rowIndex + m, colIndex - m - 1) - integral_ssd.at<int>(rowIndex - m - 1, colIndex + m) + integral_ssd.at<int>(rowIndex - m - 1, colIndex - m - 1);
				int ssd = getMatElement(integral_ssd, rowIndex + m, colIndex + m) - getMatElement(integral_ssd, rowIndex + m, colIndex - m - 1) - getMatElement(integral_ssd, rowIndex - m - 1, colIndex + m) + getMatElement(integral_ssd, rowIndex - m - 1, colIndex - m - 1);
				if (ssd < min_ssd.at<int>(rowIndex, colIndex) && ssd != 0) {
					min_ssd.at<int>(rowIndex, colIndex) = ssd;
					original_disparitiy.at<int>(rowIndex, colIndex) = d;
				}
			}
		}
	}

	// fix the left most side
	for (int rowIndex = m; rowIndex < DisparityMap.rows - m - 1; rowIndex++) {
		for (int colIndex = m; colIndex < DISPARITY_RANGE; colIndex++) {
			original_disparitiy.at<int>(rowIndex, colIndex) = original_disparitiy.at<int>(rowIndex, DISPARITY_RANGE);
		}
	}

	normalize(original_disparitiy, original_disparitiy, 0.0, 255.0, NORM_MINMAX);
	original_disparitiy.convertTo(DisparityMap, CV_8U, 1.0, 0);
}

void matchingDP(int windowsize, Mat SrcImg1, Mat SrcImg2, Mat & DisparityMap)
{
	enum Action { none, left, upper_left, up };

	Mat original_disparitiy(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));

	int m = (windowsize - 1) / 2;

	Mat* integral_ssd = new Mat[DISPARITY_RANGE];
	for (int i = 0; i < DISPARITY_RANGE; i++) {
		integral_ssd[i] = Mat(DisparityMap.rows, DisparityMap.cols, CV_32S, Scalar(0));
	}

	for (int d = 0; d < DISPARITY_RANGE; d++) {
		for (int rowIndex = 0; rowIndex < DisparityMap.rows; rowIndex++) {
			int sum = 0;
			for (int colIndex = d; colIndex < DisparityMap.cols; colIndex++) {
				sum += pow((int)SrcImg1.at<uchar>(rowIndex, colIndex) - (int)SrcImg2.at<uchar>(rowIndex, colIndex - d), 2.0);
				if (rowIndex == 0) {
					integral_ssd[d].at<int>(rowIndex, colIndex) = sum;
				}
				else {
					integral_ssd[d].at<int>(rowIndex, colIndex) = integral_ssd[d].at<int>(rowIndex - 1, colIndex) + sum;
				}
			}
		}
	}

	Mat* cost = new Mat[SrcImg1.rows];
	for (int i = 0; i < SrcImg1.rows; ++i) {
		//cost[i] = Mat(SrcImg1.cols, SrcImg2.cols, CV_32S, Scalar(0));
		cost[i] = Mat(SrcImg1.cols, SrcImg2.cols, CV_32S, Scalar(std::numeric_limits<int>::max()));
	}

	// assign edge cost
	for (int scanline = m + 1; scanline < SrcImg1.rows - m - 1; scanline++) {
		for (int lp = m + 1; lp < SrcImg1.cols - m - 1; lp++) {
			for (int rp = m + 1; rp < SrcImg2.cols - m - 1; rp++) {
				int d = lp - rp;
				if (d >= 0 && d < DISPARITY_RANGE) {
					//cost[scanline][lp][rp] = getMatElement(integral_ssd[d], scanline + m, lp + m) - getMatElement(integral_ssd[d], scanline + m, lp - m - 1) - getMatElement(integral_ssd[d], scanline - m - 1, lp + m) + getMatElement(integral_ssd[d], scanline - m - 1, lp - m - 1);
					cost[scanline].at<int>(lp, rp) = getMatElement(integral_ssd[d], scanline + m, lp + m) - getMatElement(integral_ssd[d], scanline + m, lp - m - 1) - getMatElement(integral_ssd[d], scanline - m - 1, lp + m) + getMatElement(integral_ssd[d], scanline - m - 1, lp - m - 1);
				}
				//cout << DPtable1[scanline][lp][rp] << ' ';
			}
			//cout << endl;
		}
		//break;
	}

	Mat tmp(SrcImg1.cols, SrcImg2.cols, CV_8U, Scalar(0));
	Mat tmp2(SrcImg1.cols, SrcImg2.cols, CV_32S, Scalar(0));
	normalize(cost[m + 1], tmp2, 0.0, 255.0, NORM_MINMAX);
	tmp2.convertTo(tmp, CV_8U, 1.0, 0);
	//imshow("cost[m+1]", tmp);

	for (int scanline = m + 1; scanline < SrcImg1.rows - m - 1; scanline++) {
		Mat DPtable1 = cost[scanline];
		Mat DPtable2(SrcImg1.cols, SrcImg2.cols, CV_32S, Scalar(0));

		for (int i = m + 2; i < SrcImg1.cols - m - 1; i++) {
			int d = i - (m + 1);
			if (d >= 0 && d < DISPARITY_RANGE) {
				DPtable1.at<int>(i, m + 1) = DPtable1.at<int>(i - 1, m + 1) + cost[scanline].at<int>(i, m + 1);
				DPtable2.at<int>(i, m + 1) = up;
			}
		}

		for (int lp = m + 2; lp < SrcImg1.cols - m - 1; lp++) {
			for (int rp = m + 2; rp < SrcImg2.cols - m - 1; rp++) {
				int d = lp - rp;
				if (d >= 0 && d < DISPARITY_RANGE) {
					int min = min_(DPtable1.at<int>(lp - 1, rp - 1), DPtable1.at<int>(lp - 1, rp), DPtable1.at<int>(lp, rp - 1));
					DPtable1.at<int>(lp, rp) = min + cost[scanline].at<int>(lp, rp);
					if (min == DPtable1.at<int>(lp - 1, rp - 1)) {
						DPtable2.at<int>(lp, rp) = upper_left;
					}
					if (min == DPtable1.at<int>(lp - 1, rp)) {
						DPtable2.at<int>(lp, rp) = up;
					}
					if (min == DPtable1.at<int>(lp, rp - 1)) {
						DPtable2.at<int>(lp, rp) = left;
					}
				}
			}
		}

		/*for (int i = 0; i < SrcImg1.cols; i++) {
		for (int j = 0; j < SrcImg2.cols; j++) {
		cout << DPtable2[i][j] << ' ';
		}
		cout << endl;
		}*/

		//cout << DPtable1 << endl;

		// disparity space image
		Mat dsi(SrcImg1.cols, SrcImg2.cols, CV_8U, Scalar(0));
		normalize(DPtable1, DPtable1, 0.0, 255.0, NORM_MINMAX);
		DPtable1.convertTo(dsi, CV_8U, 1.0, 0);
		//imshow("disparity space image", dsi);
		//cout << dsi;

		Mat test(SrcImg1.cols, SrcImg2.cols, CV_8U, Scalar(0));
		Mat test2(SrcImg1.cols, SrcImg2.cols, CV_32S, Scalar(0));
		normalize(DPtable2, test2, 0.0, 255.0, NORM_MINMAX);
		test2.convertTo(test, CV_8U, 1.0, 0);
		//imshow("dp path image", test);

		int leftpixel = SrcImg1.cols - m - 2;
		int rigthpixel = SrcImg2.cols - m - 2;
		int disparity = 0;
		//cout << DPtable2.at<int>(leftpixel, rigthpixel) << endl;

		for (int i = m + 1; i < SrcImg1.cols - m - 1; i++) {
			original_disparitiy.at<int>(scanline, leftpixel) = disparity;
			if (DPtable2.at<int>(leftpixel, rigthpixel) == none) {
				if (leftpixel == m + 1)
					break;
				cout << "Something wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
				cout << "(" << leftpixel << "," << rigthpixel << ")" << endl;
			}
			if (DPtable2.at<int>(leftpixel, rigthpixel) == left) {
				--rigthpixel;
				++disparity;
			}
			if (DPtable2.at<int>(leftpixel, rigthpixel) == upper_left) {
				--leftpixel;
				--rigthpixel;
			}
			if (DPtable2.at<int>(leftpixel, rigthpixel) == up) {
				--leftpixel;
				--disparity;
			}
		}
	}

	normalize(original_disparitiy, original_disparitiy, 0.0, 255.0, NORM_MINMAX);
	original_disparitiy.convertTo(DisparityMap, CV_8U, 1.0, 0);
	//imshow("TEST", DisparityMap);
}

