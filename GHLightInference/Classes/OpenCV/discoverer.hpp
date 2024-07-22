#ifndef __DISCOVERER_HPP__
#define __DISCOVERER_HPP__

#include "common.hpp"
#include "dbscan.h"
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


class LightBarDetector {
private:
    cv::Mat image;

public:
    LightBarDetector(const cv::Mat &img)
            : image(img) {}

    Mat pointErode(Mat &binary) {
        // 查找轮廓
        vector<vector<cv::Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.size() < 2)return binary;
        // 按面积从大到小排序轮廓
        sort(contours.begin(), contours.end(),
             [](const vector<cv::Point> &c1, const vector<cv::Point> &c2) {
                 return contourArea(c1) > contourArea(c2);
             });

        Mat kernelErodeMin = getStructuringElement(MORPH_ELLIPSE, cv::Size(3, 1));
        LOGD(LOG_TAG, "binary %d - %d", binary.rows, binary.cols);
        // 对面积大于200的轮廓进行腐蚀操作
        Mat dst = cv::Mat::zeros(binary.size(), CV_8UC1);
        bool hasErode = false;
        bool hasBig = false;
        // 对大于200像素的轮廓区域进行腐蚀操作
        for (size_t i = 0; i < contours.size(); i++) {
            // 计算轮廓面积
            double contourArea = cv::contourArea(contours[i]);
            if (contourArea >= 300) {
                hasBig = true;
                cv::drawContours(dst, contours, static_cast<int>(i), cv::Scalar::all(255), -1);
            } else {
                if (!hasErode && hasBig) {
                    hasErode = true;
                    erode(dst, dst, kernelErodeMin);
                }
                cv::drawContours(dst, contours, static_cast<int>(i), cv::Scalar::all(255), -1);
            }
        }
        return dst;
    }


    cv::Mat erodeElongatedContours(const cv::Mat &input) {
        cv::Mat result = input.clone();

        // 1. 找到所有轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 2. 遍历轮廓,找出长条形轮廓
        for (const auto &contour: contours) {
            // 计算轮廓的最小外接矩形
            cv::RotatedRect rect = cv::minAreaRect(contour);

            // 判断是否为长条形(可以根据需要调整比例)
            float max = std::max(rect.size.width, rect.size.height);
            float min = std::min(rect.size.width, rect.size.height);
            float ratio = max / min;
            LOGD(LOG_TAG, "ratio=%f   %f-%f", ratio, rect.size.width, rect.size.height);
            if (ratio < 4.0 && max > 50 && min > 5) {  // 假设长宽比大于3的为长条形
                // 3. 对长条形轮廓进行定向腐蚀
                cv::Point2f vertices[4];
                rect.points(vertices);

                // 计算长轴方向
                cv::Point2f direction = vertices[1] - vertices[0];
                float length = cv::norm(direction);
                direction /= length;

                // 沿长轴方向进行腐蚀
                for (float t = 0; t < length; t += 5.0) {  // 每隔5个像素放置一个点
                    cv::Point2f pt = vertices[0] + direction * t;
                    cv::circle(result, pt, 1, cv::Scalar(255), -1);
                }

                // 4. 移除原始轮廓
                cv::drawContours(result, std::vector<std::vector<cv::Point>>{contour}, 0,
                                 cv::Scalar(0), -1);
            }
        }

        return result;
    }

    cv::Mat
    adaptiveThreshold(const cv::Mat &src, int initialThreshold = 155, int minContourArea = 600,
                      int maxIterations = 8) {
        cv::Mat result = src.clone();
        cv::Mat binary;
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

        for (int i = 0; i < maxIterations; ++i) {
            // 应用二值化
            cv::threshold(result, binary, initialThreshold, 255, cv::THRESH_BINARY);

            // 查找轮廓
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // 对大面积区域应用更高的阈值
            cv::Mat roi;
            // 遍历大面积轮廓
            for (const auto &contour: contours) {
                double area = cv::contourArea(contour);
                if (i == maxIterations - 2)minContourArea = 300;
                if (area > minContourArea) {
                    // 创建掩码
                    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
                    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0,
                                     cv::Scalar(255), -1);


                    result.copyTo(roi, mask);
                } else {
                    cv::drawContours(dst, std::vector<std::vector<cv::Point>>{contour}, 0,
                                     cv::Scalar(255), -1);
                }
            }
            result = roi;
            // 增加阈值，为下一次迭代做准备
            initialThreshold += 8;
        }

        return dst;
    }

    cv::Mat preprocessImage(const cv::Mat &src, vector<Mat> &outMats) {

        cv::Mat gray, hsv, enhanced, thresh, dst;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        // 进行直方图均衡化
        cv::Mat equalized;
        cv::equalizeHist(gray, equalized);

        // 转换到HSV色彩空间
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // 定义红色和绿色的HSV范围
        Scalar lower_red1(0, 90, 165), upper_red1(8, 255, 255);
        Scalar lower_red2(175, 90, 165), upper_red2(180, 255, 255);
        Scalar lower_green(64, 90, 165), upper_green(78, 255, 255);

        // 创建红色和绿色的掩膜
        Mat mask_red1, mask_red2, mask_green;
        inRange(hsv, lower_red1, upper_red1, mask_red1);
        inRange(hsv, lower_red2, upper_red2, mask_red2);
        inRange(hsv, lower_green, upper_green, mask_green);

        // 合并红色掩膜
        Mat mask_red;
        bitwise_or(mask_red1, mask_red2, mask_red);

        // 合并红色和绿色掩膜
        Mat mask;
        bitwise_or(mask_red, mask_green, mask);

        Mat test1 = adaptiveThreshold(equalized);

//        outMats.push_back(test1);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(test1, thresh, cv::MORPH_DILATE, kernel);

        cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));


        outMats.push_back(thresh);

//        outMats.push_back(mask);

        Mat maskMin = pointErode(mask);
        cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
        cv::morphologyEx(maskMin, maskMin, cv::MORPH_CLOSE, kernel3);
        cv::morphologyEx(maskMin, maskMin, cv::MORPH_OPEN, kernel2);

        outMats.push_back(maskMin);

        bitwise_and(maskMin, thresh, dst);
//        outMats.push_back(dst);

        Mat dst2;
        cv::Mat kernel4 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1));
        cv::morphologyEx(dst, dst2, cv::MORPH_DILATE, kernel4, cv::Point(-1, -1), 2);
        outMats.push_back(thresh);
        return dst2;
    }

};

RotatedRect fitRotatedRect(const vector<LightBead> &beads);

double calculateLightBarConfidence(const LightBar &bar);

vector<LightBar> clusterBeadsIntoLightBars(const vector<LightBead> &beads);

bool isBeadAligned(const vector<LightBead> &bar, const LightBead &bead);

/**
 * 根据轮廓识别点位
 */
void
findByContours(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints, int icNum,
               vector<Mat> &outMats);


void findNoodleLamp(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints,
                    vector<Mat> &outMats);


double distanceP(Point2f p1, Point2f p2);

/**
 * 二值化灯珠轮廓开运算和腐蚀处理
 */
Mat morphologyImage(Mat &image, int openKernelSize, int dilateKernelSize, int shape);

Mat thresholdNoodleLamp(Mat &src, vector<LightPoint> &lightPoints,
                        vector<Mat> &outMats);

std::vector<std::vector<cv::Point>>
removeLargeContours(const std::vector<std::vector<cv::Point>> &contours, double threshold = 2.5,
                    size_t minContourCount = 6);

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
