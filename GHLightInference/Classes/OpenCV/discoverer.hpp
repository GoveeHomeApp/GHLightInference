#ifndef __DISCOVERER_HPP__
#define __DISCOVERER_HPP__

#include "common.hpp"
#include "dbscan.h"

using namespace cv;
using namespace std;

struct ContourInfo {
    vector<Point2i> contour;
    Point2f center;
    double area;
    RotatedRect minRect;
};

/**
 * 根据轮廓识别点位
 */
void
findByContours(Mat &image, vector<cv::Point> &pointVector, vector<LightPoint> &lightPoints, int icNum,
               vector<Mat> &outMats);

void findNoodleLamp(Mat &image, vector<cv::Point> &pointVector, vector<LightPoint> &lightPoints,
                    vector<Mat> &outMats);

Mat removeLineContours(const Mat &binary);

double distanceP(cv::Point p1, cv::Point p2);

/**
 * 二值化灯珠轮廓开运算和腐蚀处理
 */
Mat morphologyImage(Mat &image, int openKernelSize, int dilateKernelSize, int shape);

Mat thresholdNoodleLamp(Mat &src, vector<cv::Point> &pointVector, vector<LightPoint> &lightPoints,
                        vector<Mat> &outMats);

Mat thresholdNoodleLamp2(Mat &src, Mat &hue, vector<Mat> &outMats);

/**
 * 轮廓大到小排序
 */
bool compareContourAreas(vector<cv::Point> contour1, vector<cv::Point> contour2);

/**
 * 二值化得出点位
 */
Mat thresholdPoints(Mat &src, Mat &bgrSrc, Mat &hue, int color, vector<Mat> &outMats);

void polyContours(vector<Point2i> &pointVector, vector<ContourInfo> &groups, int k,
                  double stddevThreshold);

/**
 * 离群点检测
 */
vector<int>
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
