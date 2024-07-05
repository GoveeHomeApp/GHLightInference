#ifndef __INTERPOLATE682X_HPP__
#define __INTERPOLATE682X_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>
#include <map>

using namespace cv;
using namespace std;
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
#endif
