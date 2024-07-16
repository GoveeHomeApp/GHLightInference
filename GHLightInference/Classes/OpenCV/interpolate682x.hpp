#ifndef __INTERPOLATE682X_HPP__
#define __INTERPOLATE682X_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>
#include <map>

using namespace cv;
using namespace std;
enum class FitType2D {
    LINEAR_2D,
    QUADRATIC_2D,
    CUBIC_2D
};

/**
 * 获取2个端点
 */
pair<Point2f, Point2f> getEndPoints(const RotatedRect &rect);

/**
 * 2点距离
 */
float distance(const Point2f &p1, const Point2f &p2);

/**
 * 纠正点到屏幕内
 */
Point2f adjustPointToImageBoundary(const Point2f &point, const Size &imageSize);

LightPoint adjustRectToImageBoundary(const LightPoint &rect, const Size &imageSize);

/**
 * 补全点位
 */
vector<LightPoint> completeRects(const vector<LightPoint> &existingRects,
                                 int totalCount, float targetWidth, float targetHeight,
                                 const Size &imageSize);

vector<LightPoint> interpolateAndExtrapolatePoints(
        const vector<LightPoint> &input,
        int min,
        int max,
        int fitPoints, float targetWidth, float targetHeight,
        FitType2D fitType = FitType2D::LINEAR_2D
);

#endif
