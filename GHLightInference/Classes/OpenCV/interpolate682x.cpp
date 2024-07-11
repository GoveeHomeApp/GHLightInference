//
//
//
/**
 * Created by linpeng on 2024/7/4.
 *
对已知的矩形按序号排序。
遍历所有可能的序号，如果缺失，则进行插值。
插值使用线性插值方法，考虑了中心点、大小和角度
 */
#include "interpolate682x.hpp"


float distance(const Point2f &p1, const Point2f &p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

//bool isPointInImage(const Point2f &point, const Mat &image) {
//    return point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows;
//}
//
//Point2f adjustPointToImageBoundary(const Point2f &point, const Mat &image) {
//    return Point2f(
//            max(0.0f, min((float) image.cols - 1, point.x)),
//            max(0.0f, min((float) image.rows - 1, point.y))
//    );
//}
//
//
//Point2f getMidpoint(const Point2f &p1, const Point2f &p2) {
//    return (p1 + p2) * 0.5f;
//}

pair<Point2f, Point2f> getEndPoints(const RotatedRect &rect) {
    Point2f vertices[4];
    rect.points(vertices);
    Point2f topCenter = (vertices[3] + vertices[0]) * 0.5f;
    Point2f bottomCenter = (vertices[1] + vertices[2]) * 0.5f;
    return {topCenter, bottomCenter};
}

Point2f adjustPointToImageBoundary(const Point2f &point, const Size &imageSize) {
    return Point2f(
            max(0.0f, min(static_cast<float>(imageSize.width), point.x)),
            max(0.0f, min(static_cast<float>(imageSize.height), point.y))
    );
}

LightPoint adjustRectToImageBoundary(const LightPoint &rect, const Size &imageSize) {
    LightPoint adjustedRect = rect;
    adjustedRect.startPoint = adjustPointToImageBoundary(rect.startPoint, imageSize);
    adjustedRect.endPoint = adjustPointToImageBoundary(rect.endPoint, imageSize);

    Point2f center = (adjustedRect.startPoint + adjustedRect.endPoint) * 0.5f;
    float width = rect.rotatedRect.size.width;
    float height = distance(adjustedRect.startPoint, adjustedRect.endPoint);
    float angle = atan2(adjustedRect.endPoint.y - adjustedRect.startPoint.y,
                        adjustedRect.endPoint.x - adjustedRect.startPoint.x) * 180 / CV_PI;

    adjustedRect.rotatedRect = RotatedRect(center, Size2f(width, height), angle);
    return adjustedRect;
}

vector<LightPoint> completeRects(const vector<LightPoint> &existingRects,
                                 int totalCount, float targetWidth, float targetHeight,
                                 const Size &imageSize) {
    LOGD(LOG_TAG, "completeRects = %d", existingRects.size());
    map<int, LightPoint> rectMap;
    float rectLen = 0;
    for (const auto &rect: existingRects) {
        rectMap[rect.label] = rect;
        float curLen = max(rect.rotatedRect.size.width, rect.rotatedRect.size.height);
        if (rectLen < curLen)
            rectLen = curLen;
    }

    targetHeight = min(max(rectLen, targetHeight / 2), targetHeight * 2);

    vector<LightPoint> allRects;
    int prevKnownNumber = -1;
    Point2f prevKnownCenter;
    float prevKnownAngle;

    for (int i = 0; i < totalCount; ++i) {
        if (rectMap.find(i) != rectMap.end()) {
            RotatedRect rotatedRect = rectMap[i].rotatedRect;
            if (rotatedRect.size.width > rotatedRect.size.height) {
                rotatedRect.angle = rotatedRect.angle + 90;
            }
            LOGD(LOG_TAG, "纠正尺寸 = %d  angle = %f  %f - %f", i, rotatedRect.angle,
                 rotatedRect.size.width, rotatedRect.size.height);
            rotatedRect = RotatedRect(rotatedRect.center, Size2f(targetWidth, targetHeight),
                                      rotatedRect.angle);
            auto pair = getEndPoints(rotatedRect);
            // 计算起始点和终点
            rectMap[i].startPoint = pair.first;
            rectMap[i].endPoint = pair.second;
            rectMap[i].rotatedRect = rotatedRect;
            allRects.push_back(rectMap[i]);
            prevKnownNumber = i;
            prevKnownCenter = rectMap[i].rotatedRect.center;
            prevKnownAngle = rectMap[i].rotatedRect.angle;
        } else {
            LOGD(LOG_TAG, "-----》补充矩形 = %d", i);
            LightPoint newRect = LightPoint();
            newRect.label = i;
            newRect.isInterpolate = true;
            newRect.rotatedRect.size = Size2f(targetWidth, targetHeight);

            // 寻找下一个已知矩形
            int nextKnownNumber = totalCount;
            Point2f nextKnownCenter;
            float nextKnownAngle;
            for (int j = i + 1; j < totalCount; ++j) {
                if (rectMap.find(j) != rectMap.end()) {
                    nextKnownNumber = j;
                    nextKnownCenter = rectMap[j].rotatedRect.center;
                    nextKnownAngle = rectMap[j].rotatedRect.angle;
                    break;
                }
            }
            int offset = 60;
            if (i % 2 == 0) {
                offset = -60;
            }
            // 计算新矩形的位置和角度
            if (prevKnownNumber != -1 && nextKnownNumber < totalCount) {
                // 在两个已知矩形之间均匀分布
                float t = static_cast<float>(i - prevKnownNumber) /
                          (nextKnownNumber - prevKnownNumber);

                newRect.rotatedRect.center = prevKnownCenter * (1 - t) + nextKnownCenter * t;
                newRect.rotatedRect.angle = prevKnownAngle * (1 - t) + nextKnownAngle * t;
                Point2f cCenter = newRect.rotatedRect.center;
                LOGD(LOG_TAG,
                     "prevKnownNumbere= %d  nextKnownNumber = %d  t = %f",
                     prevKnownNumber, nextKnownNumber, t);
                LOGD(LOG_TAG,
                     "在两个已知矩形之间均匀分布 = %f x %f  prevCenter = %f x %f  nextCenter = %f x %f  prevKnownAngle = %f, nextKnownAngle = %f",
                     cCenter.x, cCenter.y, prevKnownCenter.x, prevKnownCenter.y, nextKnownCenter.x,
                     nextKnownCenter.y, prevKnownAngle, nextKnownAngle);
            } else if (prevKnownNumber != -1) {
                // 只有前面的已知矩形，使用等间距外推
                Point2f x = (prevKnownNumber > 0 ? rectMap[prevKnownNumber - 1].rotatedRect.center
                                                 : Point2f(0, 0));

                Point2f direction = prevKnownCenter - x;
                if (prevKnownCenter.x == 0 && prevKnownCenter.y == 0) {
                    newRect.rotatedRect.center = Point2f(x.x + 40 * (i - prevKnownNumber),
                                                         x.y + offset);
                    newRect.rotatedRect.angle = 0;
                } else {
                    newRect.rotatedRect.center =
                            prevKnownCenter + direction * (i - prevKnownNumber);
                    newRect.rotatedRect.angle = prevKnownAngle;
                }
                Point2f cCenter = newRect.rotatedRect.center;
                LOGD(LOG_TAG,
                     "prevKnownNumbere= %d  nextKnownNumber = %d  direction = %f x %f  x = %f x %f",
                     prevKnownNumber, nextKnownNumber, direction.x, direction.y, x.x, x.y);
                LOGD(LOG_TAG,
                     "只有前面的已知矩形 = %f x %f  prevCenter = %f x %f  prevKnownAngle = %f",
                     cCenter.x, cCenter.y, prevKnownCenter.x, prevKnownCenter.y, prevKnownAngle);
            } else if (nextKnownNumber < totalCount) {
                // 只有后面的已知矩形，使用等间距外推
                Point2f x = (nextKnownNumber < totalCount - 1 ? rectMap[nextKnownNumber +
                                                                        1].rotatedRect.center
                                                              : Point2f(imageSize.width,
                                                                        imageSize.height));
                Point2f direction = nextKnownCenter - x;
                if (x.x == 0 && x.y == 0) {
                    newRect.rotatedRect.center = Point2f(
                            nextKnownCenter.x - 40 * (nextKnownNumber - i),
                            nextKnownCenter.y + offset);
//                    newRect.rotatedRect.angle = nextKnownAngle + 90;
                    newRect.rotatedRect.angle = 0;
                } else {
                    newRect.rotatedRect.center =
                            nextKnownCenter - direction * (nextKnownNumber - i);
                    newRect.rotatedRect.angle = nextKnownAngle;
                }

                Point2f cCenter = newRect.rotatedRect.center;
                LOGD(LOG_TAG,
                     "prevKnownNumbere= %d  nextKnownNumber = %d  direction = %f x %f  x = %f x %f",
                     prevKnownNumber, nextKnownNumber, direction.x, direction.y, x.x, x.y);
                LOGD(LOG_TAG,
                     "只有后面的已知矩形 = %f x %f  nextKnownCenter = %f x %f  nextKnownAngle = %f",
                     cCenter.x, cCenter.y, nextKnownCenter.x, nextKnownCenter.y, prevKnownAngle);
            } else {
                // 没有任何已知矩形，使用图像中心
                newRect.rotatedRect.center = Point2f(imageSize.width / 2.0f,
                                                     imageSize.height / 2.0f);
                newRect.rotatedRect.angle = 0;
            }

            // 计算起始点和终点
            auto pair = getEndPoints(newRect.rotatedRect);
            // 计算起始点和终点
            newRect.startPoint = pair.first;
            newRect.endPoint = pair.second;
            newRect.position = newRect.rotatedRect.center;
            allRects.push_back(newRect);
        }
    }


    sort(allRects.begin(), allRects.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    return allRects;
}
