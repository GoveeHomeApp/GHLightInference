#ifndef __INTERPOLATE70CX_HPP__
#define __INTERPOLATE70CX_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

using namespace cv;
using namespace std;

struct Group {
    int start;
    int end;
    int count;
    bool isKnown;
    vector<LightPoint> points;

    Group(int s, int e, bool k) : start(s), end(e), count(e - s + 1), isKnown(k) {}
};

struct GapInfo {
    int start;
    int end;
    double ratio;
};

enum class FitType {
    LINEAR,
    QUADRATIC,
    CUBIC
};


// 使用多项式拟合进行插值
vector<LightPoint> interpolateAndExtrapolatePoints(
        const vector<LightPoint> &input,
        int maxLabel,
        int fitPoints = 20,
        FitType fitType = FitType::LINEAR
);

void
drawPolynomialPoints(cv::Mat &image, const std::vector<LightPoint> &points, const cv::Scalar &color,
                     bool drawLabels = false);

vector<LightPoint> completeLightPoints2D(const vector<LightPoint> &inputPoints, int maxNum);

vector<LightPoint> interpolatePoints3D(const vector<LightPoint> &points);

vector<GapInfo> analyzeGaps(const vector<Group> &groups, double threshold = 1.7);

vector<Group> groupLightPoints(const vector<LightPoint> &lightPoints);

/**
 * 根据相邻位置关系找出离群点
 */
void detectOutlierPoints(vector<LightPoint> &points, vector<LightPoint> &errorPoints,
                         float avgDistance);

bool canBelievePrePre(const vector<LightPoint> &points, int i, double avgDistance);

bool canBelieveNextNext(const vector<LightPoint> &points, int i, double avgDistance);

bool
canBelievedAB(Point2f A, Point2f B, const vector<LightPoint> &points, int i,
              double avgDistance);

#endif
