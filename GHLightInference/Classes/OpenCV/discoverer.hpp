#ifndef __DISCOVERER_HPP__
#define __DISCOVERER_HPP__

#include "common.hpp"
#include "dbscan.h"

using namespace cv;
using namespace std;

/**
 * 根据轮廓识别点位
 */
void findByContours(Mat &image, vector<cv::Point> &points,
                    vector<Mat> &outMats);

void findByContours2(Mat &image, vector<cv::Point> &points, vector<cv::Point> &trapezoidPoints,
                     vector<Mat> &outMats);

Mat removeLineContours(const Mat &binary);

/**
 * 二值化灯珠轮廓开运算和腐蚀处理
 */
Mat morphologyImage(Mat &image, int openKernelSize, int dilateKernelSize);

/**
 * 轮廓大到小排序
 */
bool compareContourAreas(vector<cv::Point> contour1, vector<cv::Point> contour2);

/**
 * 二值化得出点位
 */
Mat thresholdPoints(Mat &src, Mat &bgrSrc, Mat &hue, int color, vector<Mat> &outMats);

/**
 * 离群点检测
 */
vector<cv::Point>
polyPoints(vector<Point2i> &pointVector, int k, double stddevThreshold, Mat &outMat);

/**
 * 合并同一重复点位
 */
void mergePoints(vector<cv::Point> &points, double threshold);

/**
 * 获取最小等腰梯形
 */
int getMinTrapezoid(Mat &image,
                    const vector<cv::Point> &points, vector<cv::Point>
                    &trapezoid4Points);

#endif
