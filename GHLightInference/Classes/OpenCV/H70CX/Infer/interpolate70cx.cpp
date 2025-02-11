/**
 * Created by linpeng on 2024/7/4.
 *
对已知的矩形按序号排序。
遍历所有可能的序号，如果缺失，则进行插值。
插值使用线性插值方法，考虑了中心点、大小和角度
 */
#include "interpolate70cx.hpp"

double sigmoid(double x, double scale) {
    return 1.0 / (1.0 + exp(-x / scale));
}

double smoothLimit(double value, double min, double max, double transitionRange) {
    if (value > min && value < max) {
        return value;
    }
    double range = max - min;
    double scaledValue = (value - min) / range;
    double smoothedValue = sigmoid(scaledValue * 2 - 1);
    return min + smoothedValue * range * (1 - 2 * transitionRange) + range * transitionRange;
}

Point2f
extrapolatePoint(const vector<Point2f> &points, int labelDiff, FitType fitType,
                 Size sizeLimit = Size(940, 1200)) {
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

        extrapolated_x = smoothLimit(extrapolated_x, 0, static_cast<double>(sizeLimit.width - 1));
        extrapolated_y = smoothLimit(extrapolated_y, 0, static_cast<double>(sizeLimit.height - 1));
        return Point2f(extrapolated_x, extrapolated_y);
    } catch (...) {
        LOGE(LOG_TAG, "extrapolatePoint error");
        return Point2f(0, 0);
    }
}

vector<LedPoint> interpolateAndExtrapolatePoints(
        const vector<LedPoint> &input,
        int max,
        int fitPoints,
        FitType fitType
) {
    int maxLabel = max - 1;
    vector<LedPoint> result;
    unordered_set<int> existingLabels;

    // 首先，添加所有输入点并记录它们的标签
    for (const auto &point: input) {
        result.push_back(point);
        existingLabels.insert(point.id);
    }
    LOGD(LOG_TAG, "interpolateAndExtrapolatePoints %d   result= %d", maxLabel, result.size());
//    if (result.empty()) {
//        return result;
//    }
    // 对结果进行排序
    sort(result.begin(), result.end(), [](const LedPoint &a, const LedPoint &b) {
        return a.id < b.id;
    });

    // 插值
    vector<LedPoint> interpolated;
    try{
        for (size_t i = 0; i < result.size() - 1; ++i) {
            // 如果result为空，这里会导致无符号整数下溢
            // 建议: 添加对result.size()的检查
            if (result.size() > 1) {
                int start_label = result[i].id;
                int end_label = result[i + 1].id;
                int gap = end_label - start_label;

                if (gap > 1) {
                    Point2f start_point = result[i].position;
                    Point2f end_point = result[i + 1].position;
                    Point2f step = (end_point - start_point) / static_cast<float>(gap);

                    for (int j = 1; j < gap; ++j) {
                        int new_label = start_label + j;
                        if (existingLabels.find(new_label) == existingLabels.end()) {
                            Point2f new_point = start_point + step * static_cast<float>(j);

                            LedPoint lp = LedPoint();
                            lp.id = new_label;
                            lp.position = new_point;
                            interpolated.emplace_back(lp);
                            existingLabels.insert(new_label);
                        }
                    }
                }
            }
        }

        // 将插值点添加到结果中
        result.insert(result.end(), interpolated.begin(), interpolated.end());

        LOGD(LOG_TAG, "补充内点：result = %d  input = %d  补充：%d  interpolated=%d", result.size(),
             input.size(),
             result.size() - input.size(), interpolated.size());

        sort(result.begin(), result.end(), [](const LedPoint &a, const LedPoint &b) {
            return a.id < b.id;
        });
    }catch (...){
        LOGE(LOG_TAG, "补中点问题");
    }
    // 外推
    try {
        if (!result.empty()) {
            // 前向外推
//            LOGD(LOG_TAG, "补充外推： start = %d", result.front().label);
            if (result.front().id > 0) {
                vector<cv::Point2f> points;
                for (int i = 0; i < std::min(fitPoints, static_cast<int>(result.size())); ++i) {
                    points.push_back(result[i].position);
                }

                for (int i = result.front().id - 1; i >= 0; --i) {
                    if (existingLabels.find(i) == existingLabels.end()) {
                        Point2f extrapolatedPoint = extrapolatePoint(points,
                                                                     result.front().id - i,
                                                                     fitType);
                        if (extrapolatedPoint.x == 0 && extrapolatedPoint.y == 0) {
                            continue;
                        }

                        LedPoint lp = LedPoint();
                        lp.id = i;
                        lp.position = extrapolatedPoint;
                        LOGD(LOG_TAG, "1 外推： label = %d  position= %f - %f", lp.id,
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

            LOGD(LOG_TAG, "2 补充外推： end = %d  maxLabel= %d", result.back().id, maxLabel);
            // 后向外推
            if (result.back().id < maxLabel) {
                vector<cv::Point2f> points;
                for (int i = std::max(0, static_cast<int>(result.size()) - fitPoints);
                     i < result.size(); ++i) {
                    points.push_back(result[i].position);
                }

                for (int i = result.back().id + 1; i <= maxLabel; ++i) {
                    if (existingLabels.find(i) == existingLabels.end()) {
                        Point2f extrapolatedPoint = extrapolatePoint(points,
                                                                     i - result.back().id,
                                                                     fitType);
                        LedPoint lp = LedPoint();
                        lp.id = i;
                        lp.position = extrapolatedPoint;
//                        LOGD(LOG_TAG, "外推： id = %d  position= %f - %f", lp.id,
//                             lp.position.x,
//                             lp.position.y);
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
    } catch (...) {
        LOGE(LOG_TAG, "外推 error");
    }
    // 最后再次对结果进行排序
    sort(result.begin(), result.end(), [](const LedPoint &a, const LedPoint &b) {
        return a.id < b.id;
    });
    LOGD(LOG_TAG, "线性补充：result = %d  input = %d  补充：%d", result.size(), input.size(),
         result.size() - input.size());
    return result;
}


void
drawPolynomialPoints(cv::Mat &image, const vector<LightPoint> &points, const cv::Scalar &color,
                     bool drawLabels) {
    for (const auto &lp: points) {
        cv::circle(image, lp.position, 3, color, -1);
        if (drawLabels) {
            cv::putText(image, to_string(lp.label), lp.position + cv::Point2f(5, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
}

//int main() {
//    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);  // 创建一个黑色图像
//
////    vector<LightPoint> originalPoints = generateSamplePoints();
//    int maxLabel = 0;
//    for (const auto &lp: originalPoints) {
//        maxLabel = max(maxLabel, lp.label);
//    }
//
//    // 绘制原始点
//    drawPoints(image, originalPoints, cv::Scalar(0, 0, 255), true);  // 红色，显示标签
//
//    // 使用多项式拟合补全点
//    vector<LightPoint> polyPoints = interpolatePolynomial(originalPoints, maxLabel);
//    drawPoints(image, polyPoints, cv::Scalar(0, 255, 0));  // 绿色
//
//    // 使用样条插值补全点
//    vector<LightPoint> splinePoints = interpolateSpline(originalPoints, maxLabel);
//    drawPoints(image, splinePoints, cv::Scalar(255, 0, 0));  // 蓝色
//
//    cv::imshow("Interpolation Result", image);
//    cv::imwrite("interpolation_result.png", image);
//    cv::waitKey(0);
//
//    return 0;
//}

// 绘制点和曲线
void drawPointsAndCurve(cv::Mat &image, const vector<LightPoint> &originalPoints,
                        const vector<cv::Point2f> &interpolatedPoints,
                        const cv::Scalar &color) {
    // 绘制原始点
    for (const auto &lp: originalPoints) {
        cv::circle(image, lp.position, 5, cv::Scalar(0, 0, 255), -1);  // 红色
        cv::putText(image, to_string(lp.label), lp.position + cv::Point2f(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    // 绘制插值曲线
    for (int i = 1; i < interpolatedPoints.size(); ++i) {
        cv::line(image, interpolatedPoints[i - 1], interpolatedPoints[i], color, 2);
    }
}

vector<LightPoint>
completeLightPoints2D(const vector<LightPoint> &inputPoints, int maxNum) {
    vector<LightPoint> completedPoints;
    int oldSize = inputPoints.size();
    // 按lightIndex对输入点进行排序
    vector<LightPoint> sortedPoints = inputPoints;
    sort(sortedPoints.begin(), sortedPoints.end(),
         [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

    // 计算平均距离
    float avgDistance = 0;
    if (sortedPoints.size() > 1) {
        vector<float> distances;
        for (size_t i = 1; i < sortedPoints.size(); ++i) {
            Point2f diff = sortedPoints[i].position - sortedPoints[i - 1].position;
            distances.push_back(norm(diff));
        }
        avgDistance = accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    }

    // 找到第一个和最后一个存在的索引
    int firstIndex = sortedPoints.front().label;
    int lastIndex = sortedPoints.back().label;

    // 如果必要，将范围扩展到0和maxNum
    firstIndex = min(firstIndex, 0);
    lastIndex = max(lastIndex, maxNum);

    // 遍历所有可能的索引
    int currentIndex = firstIndex;
    auto it = sortedPoints.begin();

    while (currentIndex <= lastIndex) {
        if (it != sortedPoints.end() && it->label == currentIndex) {
            // 如果当前索引存在点，直接添加
            completedPoints.push_back(*it);
            ++it;
        } else {
            // 如果当前索引缺失点，需要进行插值或外推
            LightPoint newPoint = LightPoint();
            newPoint.label = currentIndex;

            // 找到缺失点前后最近的已存在点
            auto prevIt = lower_bound(sortedPoints.begin(), sortedPoints.end(), currentIndex,
                                      [](const LightPoint &position, int index) {
                                          return position.label < index;
                                      }) - 1;
            auto nextIt = upper_bound(sortedPoints.begin(), sortedPoints.end(), currentIndex,
                                      [](int index, const LightPoint &position) {
                                          return index < position.label;
                                      });

            if (prevIt >= sortedPoints.begin() && nextIt < sortedPoints.end()) {
                // 在两个已知点之间进行插值
                int totalMissing = nextIt->label - prevIt->label - 1;
                int currentMissing = currentIndex - prevIt->label;
                float ratio = static_cast<float>(currentMissing) / (totalMissing + 1);
                newPoint.position =
                        prevIt->position + (nextIt->position - prevIt->position) * ratio;
            } else if (prevIt >= sortedPoints.begin()) {
                // 在最后一个已知点之后进行外推
                Point2f direction = prevIt->position - (prevIt - 1)->position;
                direction = direction / norm(direction) * (avgDistance / 2);
                newPoint.position = prevIt->position + direction * (currentIndex - prevIt->label);
            } else if (nextIt < sortedPoints.end()) {
                // 在第一个已知点之前进行外推
                Point2f direction = (nextIt + 1)->position - nextIt->position;
                direction = direction / norm(direction) * (avgDistance / 2);
                newPoint.position = nextIt->position - direction * (nextIt->label - currentIndex);
            }
            LOGD(LOG_TAG, "complete ：%d  position =%f x %f", newPoint.label, newPoint.position.x,
                 newPoint.position.y);
            completedPoints.push_back(newPoint);
        }
        ++currentIndex;
    }
    LOGD(LOG_TAG, "completeLightPoints2D 补点：%d", completedPoints.size() - oldSize);
    return completedPoints;
}

vector<LightPoint> interpolatePoints3D(const vector<LightPoint> &points) {
    if (points.empty()) return {};
    int oldSize = points.size();
    vector<LightPoint> result = points;
    sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    // 计算点分布的实际高度
    float minY = numeric_limits<float>::max();
    float maxY = numeric_limits<float>::lowest();
    for (const auto &position: result) {
        minY = min(minY, position.position.y);
        maxY = max(maxY, position.position.y);
    }
    float distributionHeight = maxY - minY;
    float sectionHeight = distributionHeight / 5.0f;

    vector<LightPoint> interpolated;

    for (size_t i = 0; i < result.size() - 1; ++i) {
        interpolated.push_back(result[i]);
        int gap = result[i + 1].label - result[i].label - 1;

        // 确定当前点所在的区域
        int section = min(4, static_cast<int>((result[i].position.y - minY) / sectionHeight));

        // 根据区域决定最大允许间隔
        int maxGap;
        if (section == 0) {
            maxGap = 1;
        } else if (section >= 1 && section <= 3) {
            maxGap = 3;
        } else {
            maxGap = 4;
        }

        if (gap > 0 && gap < maxGap) {
            for (int j = 1; j <= gap; ++j) {
                float t = static_cast<float>(j) / (gap + 1);
                Point2f newPoint = result[i].position * (1 - t) + result[i + 1].position * t;
                LightPoint lp = LightPoint();
                lp.label = result[i].label + j;
                lp.position = newPoint;
                LOGV(LOG_TAG, "3D补连续点 = %d   p = %f - %f", lp.label, newPoint.x, newPoint.y);
                result.push_back(lp);
            }
        }
    }
    interpolated.push_back(result.back());
    LOGD(LOG_TAG, "interpolatePoints3D 补点：%d", interpolated.size() - oldSize);
    return interpolated;
}


/**
首先按 label 对点进行排序。
遍历所有可能的索引，检查是否存在对应的点。
根据点的存在与否，创建已知点组或空缺组。
对于已知点组，保存其中的所有 LightPoint 对象。
对于空缺组，points 向量保持为空。
 * @param lightPoints
 * @return
 */
vector<Group> groupLightPoints(const vector<LightPoint> &lightPoints) {
    vector<Group> groups;
    if (lightPoints.empty()) return groups;

    // Sort light points by their index
    vector<LightPoint> sortedPoints = lightPoints;
    sort(sortedPoints.begin(), sortedPoints.end(),
         [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

    int start = sortedPoints[0].label;
    bool isKnown = true;
    vector<LightPoint> currentGroupPoints;

    for (int i = 0; i <= sortedPoints.back().label; ++i) {
        auto it = find_if(sortedPoints.begin(), sortedPoints.end(),
                          [i](const LightPoint &lp) { return lp.label == i; });

        bool pointExists = (it != sortedPoints.end());

        if (!pointExists || i == sortedPoints.back().label) {
            if (isKnown || i == sortedPoints.back().label) {
                Group group(start, i - 1, isKnown);
                group.points = currentGroupPoints;
                groups.push_back(group);
            }

            if (!pointExists && i < sortedPoints.back().label) {
                start = i;
                isKnown = false;
                currentGroupPoints.clear();
            }
        }

        if (pointExists) {
            if (!isKnown) {
                if (start < i - 1) {
                    groups.emplace_back(start, i - 1, false);
                }
                start = i;
                isKnown = true;
                currentGroupPoints.clear();
            }
            currentGroupPoints.push_back(*it);
        }
    }

    /**
     
struct Group {
    int start;
    int end;
    int count;
    bool isKnown;
    vector<LightPoint> points;

    Group(int s, int e, bool k) : start(s), end(e), count(e - s + 1), isKnown(k) {}
};
     */
//    for (const auto &item: groups) {
//        int isKnownInt = 0;
//        if (item.isKnown)isKnownInt = 1;
//        LOGD(LOG_TAG, "Group: start=%d   end=%d   count=%d    isKnown=%d  ", item.start, item.end,
//             item.count, item.isKnown);
//    }
    return groups;
}

/**
遍历所有组，累计连续的已知组的点数（kNum）。
当遇到缺失组时，比较缺失组的点数（ukNum）与之前累计的已知组点数。
如果 ukNum > threshold * kNum，则记录这个缺失组的信息。
每次遇到缺失组后重置 kNum，为下一组已知组做准备。
 * @param groups
 * @param threshold
 * @return
 */
vector<GapInfo> analyzeGaps(const vector<Group> &groups, double threshold) {
    vector<GapInfo> significantGaps;
    int kNum = 0;

    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].isKnown) {
            kNum += groups[i].count;
        } else {
            int ukNum = groups[i].count;
            if (kNum > 0 && static_cast<double>(ukNum) > threshold * kNum) {
                significantGaps.push_back(
                        {groups[i].start, groups[i].end, static_cast<double>(ukNum) / kNum});
            }
            kNum = 0;  // 重置 kNum，为下一个非缺失组做准备
        }
    }
    /**
struct GapInfo {
    int start;
    int end;
    double ratio;
};
     */
    for (const auto &item: significantGaps) {
        LOGD(LOG_TAG, "缺失大组: start=%d   end=%d   ratio=%f ", item.start, item.end, item.ratio);
    }
    return significantGaps;
}

/**
 * 判断位置i的点的前3个点分布是否正常
 * @return
 */
bool canBelievePrePre(const vector<LightPoint> &points, int i, double avgDistance) {
    try {
        if (i >= 2 && abs(points[i].label - points[i - 1].label) < 4) {
            float distPrev = norm(points[i].position - points[i - 1].position);
            float distancePrev = abs(points[i].label - points[i - 1].label) * avgDistance * 1.5;

            float distPrevPrev = norm(points[i - 1].position - points[i - 2].position);
            float distancePrevPrev =
                    abs(points[i - 1].label - points[i - 2].label) * avgDistance * 1.5;
            bool isOutlier = (distPrev > distancePrev) && (distPrevPrev <= distancePrevPrev);
            //前面至少存在2个间距是合理的情况下，判断当前点位可能合理
            return !isOutlier;
        }
    } catch (...) {
        LOGE(LOG_TAG, "canBelievePrePre error");
        return false;
    }
    return false;
}

/**
 * 判断位置i的点的后3个点分布是否正常
 * @return
 */
bool canBelieveNextNext(const vector<LightPoint> &points, int i, double avgDistance) {
    try {
        if (i < points.size() - 2 && abs(points[i].label - points[i - 1].label) > 4 &&
            abs(points[i].label - points[i + 1].label) < 4) {
            float distanceNext = abs(points[i].label - points[i + 1].label) * avgDistance * 1.5;
            float distNext = norm(points[i].position - points[i + 1].position);
            float distNextNext = norm(points[i + 1].position - points[i + 2].position);
            float distanceNextNext =
                    abs(points[i + 1].label - points[i + 2].label) * avgDistance * 1.5;
            bool isOutlier = (distNext > distanceNext) &&
                             (distNextNext <= distanceNextNext);
            return !isOutlier;
        }
    } catch (...) {
        LOGE(LOG_TAG, "canBelieveNextNext error");
        return false;
    }
    return false;
}

/**
 * 判断A B2个点是否是能够相信的
 */
bool
canBelievedAB(Point2f start, Point2f end, const vector<LightPoint> &points, int i,
              double avgDistance) {
    //i 推断点的下一个点
    float distance = norm(start - end);
    if (distance > avgDistance * 1.5) {
        LOGD(LOG_TAG, "canBelievedAB 当前点距离过大，i-label = %d", points[i].label);
        return false;
    }
    bool believePrePre = canBelievePrePre(points, i, avgDistance);
    if (believePrePre) return believePrePre;
    return canBelieveNextNext(points, i, avgDistance);
}


/**
 * 根据相邻位置关系找出离群点
 */
void detectOutlierPoints(vector<LightPoint> &points, vector<LightPoint> &errorPoints,
                         float avgDistance) {
    try {
        int n = points.size();
        LOGW(LOG_TAG, "-----detectOutlierPoints");
        if (n < 5) return;  // Need at least 5 points for this algorithm

        sort(points.begin(), points.end(),
             [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

        for (int i = n - 3; i >= 2; --i) {
            //当前点的2侧都有点
            if (abs(points[i].label - points[i - 1].label) < 4 &&
                abs(points[i].label - points[i + 1].label) < 4) {

                float distPrev = norm(points[i].position - points[i - 1].position);
                float distNext = norm(points[i].position - points[i + 1].position);
                float distPrevPrev = norm(points[i - 1].position - points[i - 2].position);
                float distNextNext = norm(points[i + 1].position - points[i + 2].position);
                float distSkip = norm(points[i - 1].position - points[i + 1].position);

                float distancePrev = abs(points[i].label - points[i - 1].label) * avgDistance * 2.2;
                float distanceNext = abs(points[i].label - points[i + 1].label) * avgDistance * 2.2;
                float distanceSkip =
                        abs(points[i - 1].label - points[i + 1].label) * avgDistance * 1.5;
                float distancePrevPrev =
                        abs(points[i - 1].label - points[i - 2].label) * avgDistance * 1.5;
                float distanceNextNext =
                        abs(points[i + 1].label - points[i + 2].label) * avgDistance * 1.5;
                bool isOutlier = (distPrev > distancePrev || distNext > distanceNext) &&
                                 (distSkip <= distanceSkip) &&
                                 (distPrevPrev <= distancePrevPrev ||
                                  distNextNext <= distanceNextNext);
                if (isOutlier) {
                    LOGE(LOG_TAG, "1---揪出离群点：%d", points[i].label);
                    errorPoints.push_back(points[i]);
                    points.erase(points.begin() + i);
                }
            } else if (abs(points[i].label - points[i - 1].label) > 4 &&
                       abs(points[i].label - points[i + 1].label) < 4) {

                //与下一个点的阀值距离
                float distanceNext = abs(points[i].label - points[i + 1].label) * avgDistance * 2.2;
                //与下一个点的距离
                float distNext = norm(points[i].position - points[i + 1].position);
                //后续2个点的距离
                float distNextNext = norm(points[i + 1].position - points[i + 2].position);
                //与下下一个点的阀值距离
                float distanceNextNext =
                        abs(points[i + 1].label - points[i + 2].label) * avgDistance * 1.5;

                bool isOutlier = (distNext > distanceNext) && (distNextNext <= distanceNextNext);
                if (isOutlier) {
                    LOGE(LOG_TAG, "2---揪出离群点：%d", points[i].label);
                    errorPoints.push_back(points[i]);
                    points.erase(points.begin() + i);
                } else if (abs(points[i].label - points[i - 1].label) > 15 &&
                           distNextNext > distanceNextNext * 2) {
                    LOGE(LOG_TAG, "2个点都是离群点 当前label = %d", points[i].label);
                    errorPoints.push_back(points[i]);
                    points.erase(points.begin() + i);
                }
            } else if (abs(points[i].label - points[i - 1].label) < 4 &&
                       abs(points[i].label - points[i + 1].label) > 4) {
                float distPrev = norm(points[i].position - points[i - 1].position);
                float distancePrev = abs(points[i].label - points[i - 1].label) * avgDistance * 2.2;

                float distPrevPrev = norm(points[i - 1].position - points[i - 2].position);
                float distancePrevPrev =
                        abs(points[i - 1].label - points[i - 2].label) * avgDistance * 1.5;
                bool isOutlier = (distPrev > distancePrev) && (distPrevPrev <= distancePrevPrev);

                if (isOutlier) {
                    LOGE(LOG_TAG, "3---揪出离群点：%d", points[i].label);
                    errorPoints.push_back(points[i]);
                    points.erase(points.begin() + i);
                }
            } else {

            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "detectOutlierPoints error");
    }
}
