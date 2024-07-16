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

cv::Point2f
extrapolatePoint(const std::vector<cv::Point2f> &points, int labelDiff, FitType2D fitType) {
    if (points.size() < 2) return cv::Point2f(0, 0);  // Not enough points to extrapolate

    std::vector<double> x, y;
    for (const auto &p: points) {
        x.push_back(p.x);
        y.push_back(p.y);
    }

    cv::Mat A, B, coeffs;
    int degree = static_cast<int>(fitType);

    // Prepare matrices for polynomial fitting
    A = cv::Mat::zeros(x.size(), degree + 1, CV_64F);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j <= degree; ++j) {
            A.at<double>(i, j) = std::pow(x[i], j);
        }
    }

    cv::Mat y_mat(y);
    cv::solve(A, y_mat, coeffs, cv::DECOMP_QR);

    // Extrapolate
    double extrapolated_x = x.back() + labelDiff * (x.back() - x.front()) / (points.size() - 1);
    double extrapolated_y = 0;
    for (int i = 0; i <= degree; ++i) {
        extrapolated_y += coeffs.at<double>(i) * std::pow(extrapolated_x, i);
    }

    return cv::Point2f(extrapolated_x, extrapolated_y);
}

vector<LightPoint> interpolateAndExtrapolatePoints(
        const vector<LightPoint> &input,
        int min,
        int max,
        int fitPoints, float targetWidth, float targetHeight,
        FitType2D fitType
) {
    int maxLabel = max - 1;
    vector<LightPoint> result;
    unordered_set<int> existingLabels;
    float rectLen = 0;
    for (const auto &rect: input) {
        float curLen = cv::max(rect.rotatedRect.size.width, rect.rotatedRect.size.height);
        if (rectLen < curLen)
            rectLen = curLen;
    }

    targetHeight = cv::min(cv::max(rectLen, targetHeight / 2), targetHeight * 2);

    // 首先，添加所有输入点并记录它们的标签
    for (size_t k = 0; k < input.size(); ++k) {
        LightPoint lk = input[k];
        if (lk.rotatedRect.size.width > lk.rotatedRect.size.height) {
            lk.rotatedRect = RotatedRect(lk.rotatedRect.center, Size2f(targetHeight, targetWidth),
                                         lk.rotatedRect.angle);
        } else {
            lk.rotatedRect = RotatedRect(lk.rotatedRect.center, Size2f(targetWidth, targetHeight),
                                         lk.rotatedRect.angle);
        }
        auto pair = getEndPoints(lk.rotatedRect);
        lk.startPoint=pair.first;
        lk.endPoint=pair.second;
        result.push_back(lk);
        existingLabels.insert(lk.label);
    }

    LOGD(LOG_TAG, "interpolateAndExtrapolatePoints %d   result= %d", maxLabel, result.size());
    // 对结果进行排序
    std::sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    // 插值
    std::vector<LightPoint> interpolated;
    for (size_t i = 0; i < result.size() - 1; ++i) {
        int start_label = result[i].label;
        int end_label = result[i + 1].label;
        int gap = end_label - start_label;

        if (gap > 1) {
            cv::Point2f start_point = result[i].position;
            cv::Point2f end_point = result[i + 1].position;
            cv::Point2f step = (end_point - start_point) / static_cast<float>(gap);

            for (int j = 1; j < gap; ++j) {
                int new_label = start_label + j;
                if (existingLabels.find(new_label) == existingLabels.end()) {
                    cv::Point2f new_point = start_point + step * static_cast<float>(j);

                    LightPoint lp = LightPoint();
                    lp.label = new_label;
                    lp.position = new_point;
                    lp.rotatedRect = RotatedRect(new_point, Size(result[i].with, result[i].height),
                                                 result[i].rotatedRect.angle);
                    lp.tfRect = lp.rotatedRect.boundingRect();
                    lp.with = result[i].with;
                    lp.height = result[i].height;
                    auto pair = getEndPoints(lp.rotatedRect);
                    lp.startPoint=pair.first;
                    lp.endPoint=pair.second;
                    interpolated.emplace_back(lp);
                    LOGD(LOG_TAG, "插入： %d", new_label);
                    existingLabels.insert(new_label);
                }
            }
        }
    }

    // 将插值点添加到结果中
    result.insert(result.end(), interpolated.begin(), interpolated.end());

    LOGD(LOG_TAG, "补充内点：result = %d  input = %d  补充：%d  interpolated=%d", result.size(),
         input.size(),
         result.size() - input.size(), interpolated.size());

    std::sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    // 外推
    if (!result.empty()) {
        // 前向外推
        LOGD(LOG_TAG, "补充外推： start = %d  min = %d", result.front().label, min);
        if (result.front().label > min) {
            vector<cv::Point2f> points;
            for (int i = 0; i < cv::min(fitPoints, static_cast<int>(result.size())); ++i) {
                points.push_back(result[i].position);
            }

            for (int i = result.front().label - 1; i >= 0; --i) {
                if (existingLabels.find(i) == existingLabels.end()) {
                    cv::Point2f extrapolatedPoint = extrapolatePoint(points,
                                                                     result.front().label - i,
                                                                     fitType);
                    LightPoint lp = LightPoint();
                    lp.label = i;
                    lp.position = extrapolatedPoint;
                    RotatedRect rotatedRect;
                    LightPoint reference = result.front();
                    rotatedRect = RotatedRect(extrapolatedPoint,
                                              Size2f(reference.rotatedRect.size.width,
                                                     reference.rotatedRect.size.height),
                                              reference.rotatedRect.angle);

                    lp.rotatedRect = rotatedRect;
                    lp.tfRect = lp.rotatedRect.boundingRect();
                    lp.with = reference.with;
                    lp.height = reference.height;
                    auto pair = getEndPoints(lp.rotatedRect);
                    lp.startPoint=pair.first;
                    lp.endPoint=pair.second;
                    LOGD(LOG_TAG, "1 外推： label = %d  position= %f - %f", lp.label, lp.position.x,
                         lp.position.y);
                    result.emplace_back(lp);
                    existingLabels.insert(i);
                    // 更新拟合点集
                    points.insert(points.begin(), extrapolatedPoint);
                    if (points.size() > fitPoints) {
                        points.pop_back();
                    }
                }
            }
        }

        LOGD(LOG_TAG, "2 补充外推： end = %d  maxLabel= %d", result.back().label, maxLabel);
        // 后向外推
        if (result.back().label < maxLabel) {
            std::vector<cv::Point2f> points;
            for (int i = std::max(0, static_cast<int>(result.size()) - fitPoints);
                 i < result.size(); ++i) {
                points.push_back(result[i].position);
            }

            for (int i = result.back().label + 1; i <= maxLabel; ++i) {
                if (existingLabels.find(i) == existingLabels.end()) {
                    cv::Point2f extrapolatedPoint = extrapolatePoint(points,
                                                                     i - result.back().label,
                                                                     fitType);
                    LightPoint lp = LightPoint();
                    lp.label = i;
                    lp.position = extrapolatedPoint;
                    RotatedRect rotatedRect;

                    LightPoint reference = result.back();

                    rotatedRect = RotatedRect(extrapolatedPoint,
                                              Size2f(reference.rotatedRect.size.width,
                                                     reference.rotatedRect.size.height),
                                              reference.rotatedRect.angle);


                    lp.rotatedRect = rotatedRect;
                    lp.tfRect = lp.rotatedRect.boundingRect();
                    lp.with = reference.with;
                    lp.height = reference.height;
                    auto pair = getEndPoints(lp.rotatedRect);
                    lp.startPoint=pair.first;
                    lp.endPoint=pair.second;
                    LOGD(LOG_TAG, "外推： label = %d  position= %f - %f", lp.label, lp.position.x,
                         lp.position.y);
                    result.emplace_back(lp);
                    existingLabels.insert(i);
                    // 更新拟合点集
                    points.push_back(extrapolatedPoint);
                    if (points.size() > fitPoints) {
                        points.erase(points.begin());
                    }
                }
            }
        }
    }

    // 最后再次对结果进行排序
    std::sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });
    LOGD(LOG_TAG, "线性补充：result = %d  input = %d  补充：%d", result.size(), input.size(),
         result.size() - input.size());
    return result;
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
