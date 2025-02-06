#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

//#include "common.hpp"
#include "discoverer.hpp"
#include <map>
#include "common.hpp"
#include "color_splitter.hpp"
#include "interpolate70cx.hpp"

using namespace cv;
using namespace std;

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat, vector<Mat> &outMats);

/**
 * 根据定义好的步骤进行灯带排序
 * @param frameStep 当前轮数
 * @param resultObjects 当前tf识别的结果以及opencv找色的结果
 * @param outMats 输出流程中的测试图像
 * @return
 */
String
sortStripByStep(int frameStep, vector<LightPoint> &resultObjects, int lightType,
                vector<Mat> &outMats);

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LedPoint> &points, int lightTypeSet);

void processInitialFrame(vector<Point2f> pPointXys, vector<Mat> &outMats);

void release();

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2f> &points);

string splicedJson(string a, string b);

Rect2i safeRect2i(const Rect2i &region, const cv::Size &imageSize);

void drawPointsWithCircles(const Mat &src, const vector<Point2f> points, vector<Mat> &outMats,
                           const string &title = "");

void drawPointsWithLabels(const Mat &src, const vector<LedPoint> lightPoints, vector<Mat> &outMats,
                          const string &title = "");

bool compareIndex(const LedPoint &p1, const LedPoint &p2);

// 计算平均圆直径
float calculateAverageCircleDiameter(const vector<LightPoint>& points);

// 移除默认参数，因为现在我们总是会计算实际的阈值
void mergeNearbyPoints(vector<Point2f>& points, float threshold);

vector<LedPoint> validatePointPositions(const vector<LedPoint>& points, float min_valid_neighbors = 2);

#endif
