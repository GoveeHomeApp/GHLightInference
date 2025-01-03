#ifndef __DISCOVERER_HPP__
#define __DISCOVERER_HPP__

#include "common.hpp"
#include <numeric>

using namespace cv;
using namespace std;

struct LightBead {
    cv::Point center;
    float radius;
};

struct LightBar {
    vector<LightBead> beads;
    RotatedRect rect;
    double confidence;
    bool bigArea = false;
};
struct ContourInfo {
    vector<cv::Point> contour;
    cv::Point center;
    double area;
};



RotatedRect fitRotatedRect(const vector<LightBead> &beads);

/**
 * 二值化灯珠轮廓开运算和腐蚀处理
 */
Mat morphologyImage(Mat &image, int openKernelSize, int dilateKernelSize, int shape);

Mat thresholdNoodleLamp(Mat &src, vector<LightPoint> &lightPoints,
                        vector<Mat> &outMats);

/**
 * 轮廓大到小排序
 */
bool compareContourAreas(vector<cv::Point> contour1, vector<cv::Point> contour2);

/**
 * 二值化得出点位
 */
Mat thresholdPoints(Mat &src, Mat &bgrSrc, Mat &hue, int color, vector<Mat> &outMats);

void polyContours(vector<cv::Point> &pointVector, vector<ContourInfo> &groups, int k,
                  double stddevThreshold);

/**
 * 离群点检测
 */
vector<int>
polyPoints(vector<Point2f> &pointVector, int k, double stddevThreshold);//, Mat &outMat

/**
 * 合并同一重复点位
 */
void mergePoints(vector<Point2f> &points, double threshold);

/**
 * 获取最小等腰梯形
 */
int getMinTrapezoid(Mat &image,
                    const vector<Point2f> &points, vector<Point2f>
                    &trapezoid4Points);

Rect2i safeRect(const cv::Rect2i &region, const cv::Size &imageSize);

#endif
