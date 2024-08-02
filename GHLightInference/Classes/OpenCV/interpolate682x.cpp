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
#include <cmath>
#include "inferredp.hpp"

float distance(const Point2f &p1, const Point2f &p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

pair<Point2f, Point2f> getEndPoints(const RotatedRect &rect) {
    Point2f vertices[4];
    rect.points(vertices);

    // Calculate edge lengths
    float edge1 = norm(vertices[1] - vertices[0]);
    float edge2 = norm(vertices[2] - vertices[1]);

    Point2f midpoint1, midpoint2;
    if (edge1 < edge2) {
        // Short edges are [0]-[1] and [2]-[3]
        midpoint1 = (vertices[0] + vertices[1]) * 0.5f;
        midpoint2 = (vertices[2] + vertices[3]) * 0.5f;
    } else {
        // Short edges are [1]-[2] and [3]-[0]
        midpoint1 = (vertices[1] + vertices[2]) * 0.5f;
        midpoint2 = (vertices[3] + vertices[0]) * 0.5f;
    }

    // Determine which midpoint is on top
    Point2f topCenter, bottomCenter;
    if (midpoint1.y < midpoint2.y) {
        topCenter = midpoint1;
        bottomCenter = midpoint2;
    } else {
        topCenter = midpoint2;
        bottomCenter = midpoint1;
    }

    return {topCenter, bottomCenter};
}


Point2f
extrapolatePoint(const vector<Point2f> &points, int labelDiff, FitType2D fitType,
                 Size sizeLimit = Size(1450, 1950)) {
    if (points.size() < 2) return Point2f(0, 0); // Not enough points to extrapolate
    try {
        vector<double> x, y;
        for (const auto &p: points) {
            x.push_back(p.x);
            y.push_back(p.y);
        }

        Mat A, coeffs;
        int degree = static_cast<int>(fitType);

        // Prepare matrices for polynomial fitting
        A = Mat::zeros(x.size(), degree + 1, CV_64F);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j <= degree; ++j) {
                A.at<double>(i, j) = pow(x[i], j);
            }
        }

        Mat y_mat(y);
        solve(A, y_mat, coeffs, DECOMP_QR);

        // Calculate the direction vector
        double dx = x.back() - x.front();
        double dy = y.back() - y.front();
        double length = sqrt(dx * dx + dy * dy);

        // Normalize the direction vector
        if (length > 1e-6) {  // Avoid division by zero
            dx /= length;
            dy /= length;
        } else {
            // If points are too close, use a default direction (e.g., positive x-axis)
            dx = 1.0;
            dy = 0.0;
        }

        // Calculate the step size
        double step = length / (points.size() - 1);

        // Extrapolate both x and y
        double extrapolated_x = x.back() + labelDiff * step * dx;
        double extrapolated_y = y.back() + labelDiff * step * dy;

        // If polynomial fitting is desired (degree > 0), adjust y using the fitted polynomial
        if (degree > 0) {
            double fitted_y = 0;
            for (int i = 0; i <= degree; ++i) {
                fitted_y += coeffs.at<double>(i) * pow(extrapolated_x, i);
            }

            // Blend the linear extrapolation with the polynomial fit
            double alpha = 0.2; // Adjust this value to control the blend
            extrapolated_y = alpha * extrapolated_y + (1 - alpha) * fitted_y;
        }

//    LOGD(LOG_TAG, "sizeLimit  = %d, %d   xtrapolatedd %f - %f", sizeLimit.width, sizeLimit.height,
//         extrapolated_x, extrapolated_y);
        // Apply smooth limiting to the extrapolated point
        extrapolated_x = smoothLimit(extrapolated_x, 50, static_cast<double>(sizeLimit.width - 1));
        extrapolated_y = smoothLimit(extrapolated_y, 50, static_cast<double>(sizeLimit.height - 1));

        return Point2f(extrapolated_x, extrapolated_y);
    } catch (...) {
        LOGE(LOG_TAG, "extrapolatePoint error");
        return Point2f(0, 0);
    }
}

vector<LightPoint> translateRotatedRects(const vector<LightPoint> &lps, float dx, float dy) {
    if (lps.empty())return lps;
    vector<LightPoint> translatedLps;
    translatedLps.reserve(lps.size());

    for (const auto &lp: lps) {
        LightPoint newLp = LightPoint(lp.label);

        RotatedRect rect = lp.rotatedRect;
        // 创建一个新的RotatedRect，只更新中心点的坐标
        Point2f newCenter(rect.center.x + dx, rect.center.y + dy);
        RotatedRect translatedRect(newCenter, rect.size, rect.angle);
        newLp.position = newCenter;
        newLp.rotatedRect = translatedRect;
        newLp.with = translatedRect.size.width;
        newLp.height = translatedRect.size.height;
        auto pair = getEndPoints(lp.rotatedRect);
        newLp.startPoint = pair.first;
        newLp.endPoint = pair.second;
        translatedLps.push_back(newLp);
    }
    return translatedLps;
}

vector<LightPoint> interpolateAndExtrapolatePoints(const Mat &src,
                                                   vector<LightPoint> &input,
                                                   int min,
                                                   int max,
                                                   int fitPoints, float targetWidth,
                                                   float targetHeight,
                                                   FitType2D fitType
) {
    int maxLabel = max - 1;
    vector<LightPoint> result;
    unordered_set<int> existingLabels;
    if (input.empty()) {
        LightPoint lp = LightPoint();
        lp.label = min;
        lp.position = Point2f(src.cols / 2, src.rows / 2);
        lp.rotatedRect = RotatedRect(lp.position, Size(targetWidth, targetHeight),
                                     90);
        lp.tfRect = lp.rotatedRect.boundingRect();
        lp.with = targetWidth;
        lp.height = targetHeight;
        auto pair = getEndPoints(lp.rotatedRect);
        lp.startPoint = pair.first;
        lp.endPoint = pair.second;
        input.push_back(lp);
    }

    float rectLen = 0;
    for (const auto &rect: input) {
        float curLen = cv::max(rect.rotatedRect.size.width, rect.rotatedRect.size.height);
        if (rectLen < curLen)
            rectLen = curLen;
    }

    targetHeight = cv::min(cv::max(rectLen, targetHeight / 2), targetHeight * 2);

    float tranW = (1500 - src.rows) / 2.0f;
    float tranH = (2000 - src.cols) / 2.0f;
    LOGD(LOG_TAG, "tranW = %f, tranH = %f  src = %d - %d", tranW, tranH, src.rows, src.cols);
    vector<LightPoint> transInput = translateRotatedRects(input, tranW, tranH);

    // 首先，添加所有输入点并记录它们的标签
    for (size_t k = 0; k < transInput.size(); ++k) {
        LightPoint lk = input[k];
        if (lk.rotatedRect.size.width > lk.rotatedRect.size.height) {
            lk.rotatedRect = RotatedRect(lk.rotatedRect.center, Size2f(targetHeight, targetWidth),
                                         lk.rotatedRect.angle);
        } else {
            lk.rotatedRect = RotatedRect(lk.rotatedRect.center, Size2f(targetWidth, targetHeight),
                                         lk.rotatedRect.angle);
        }
        auto pair = getEndPoints(lk.rotatedRect);
        lk.startPoint = pair.first;
        lk.endPoint = pair.second;
        lk.with = targetWidth;
        lk.height = targetHeight;
        result.push_back(lk);
        existingLabels.insert(lk.label);
    }

    LOGD(LOG_TAG, "interpolateAndExtrapolatePoints %d   result= %d", maxLabel, result.size());
    // 对结果进行排序
    sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    // 插值
    vector<LightPoint> interpolated;
    for (size_t i = 0; i < result.size() - 1; ++i) {
        int start_label = result[i].label;
        int end_label = result[i + 1].label;
        int gap = end_label - start_label;

        if (gap > 1) {
            Point2f start_point = result[i].position;
            Point2f end_point = result[i + 1].position;
            Point2f step = (end_point - start_point) / static_cast<float>(gap);

            for (int j = 1; j < gap; ++j) {
                int new_label = start_label + j;
                if (existingLabels.find(new_label) == existingLabels.end()) {
                    Point2f new_point = start_point + step * static_cast<float>(j);

                    LightPoint lp = LightPoint();
                    lp.label = new_label;
                    lp.position = new_point;
                    lp.rotatedRect = RotatedRect(new_point, Size(result[i].rotatedRect.size.width,
                                                                 result[i].rotatedRect.size.height),
                                                 result[i].rotatedRect.angle);
                    lp.tfRect = lp.rotatedRect.boundingRect();
                    lp.with = targetWidth;
                    lp.height = targetHeight;
                    auto pair = getEndPoints(lp.rotatedRect);
                    lp.startPoint = pair.first;
                    lp.endPoint = pair.second;
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

    sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    // 外推
    if (!result.empty()) {
        // 前向外推
        LOGD(LOG_TAG, "补充外推： start = %d  min = %d", result.front().label, min);
        try {
            if (result.front().label > min) {
                vector<Point2f> points;
                for (int i = 0; i < cv::min(fitPoints, static_cast<int>(result.size())); ++i) {
                    points.push_back(result[i].position);
                }

                for (int i = result.front().label - 1; i >= min; --i) {
                    if (existingLabels.find(i) == existingLabels.end()) {
                        //const vector<cv::Point2f>& points, int labelDiff, FitType2D fitType, cv::Size sizeLimit
                        Point2f extrapolatedPoint = extrapolatePoint(points,
                                                                     result.front().label - i,
                                                                     fitType);
                        LightPoint lp = LightPoint();
                        lp.label = i;
                        lp.position = extrapolatedPoint;
                        LightPoint reference = result.front();
                        RotatedRect rotatedRect = RotatedRect(extrapolatedPoint,
                                                              Size2f(reference.rotatedRect.size.width,
                                                                     reference.rotatedRect.size.height),
                                                              reference.rotatedRect.angle);

                        lp.rotatedRect = rotatedRect;
                        lp.tfRect = lp.rotatedRect.boundingRect();
                        lp.with = targetWidth;
                        lp.height = targetHeight;
                        auto pair = getEndPoints(lp.rotatedRect);
                        lp.startPoint = pair.first;
                        lp.endPoint = pair.second;
                        LOGD(LOG_TAG, "前向外推： label = %d  position= %f - %f", lp.label,
                             lp.position.x,
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
        } catch (...) {
            LOGE(LOG_TAG, "前向外推");
        }
        try {
            LOGD(LOG_TAG, "2 补充外推： end = %d  maxLabel= %d", result.back().label, maxLabel);
            // 后向外推
            if (result.back().label < maxLabel) {
                vector<Point2f> points;
                for (int i = 0; i < result.size(); ++i) {
                    LOGD(LOG_TAG, "points push %d", result[i].label);
                    points.push_back(result[i].position);
                }

                for (int i = result.back().label + 1; i <= maxLabel; ++i) {
                    if (existingLabels.find(i) == existingLabels.end()) {
                        Point2f extrapolatedPoint = extrapolatePoint(points,
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
                        lp.with = targetWidth;
                        lp.height = targetHeight;
                        auto pair = getEndPoints(lp.rotatedRect);
                        lp.startPoint = pair.first;
                        lp.endPoint = pair.second;
                        LOGD(LOG_TAG, "后向外推： label = %d  position= %f - %f", lp.label,
                             lp.position.x,
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
        } catch (...) {
            LOGE(LOG_TAG, "后向外推");
        }
    }

    // 最后再次对结果进行排序
    sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });
    LOGD(LOG_TAG, "线性补充：result = %d  input = %d  补充：%d", result.size(), input.size(),
         result.size() - input.size());
    return result;
}
