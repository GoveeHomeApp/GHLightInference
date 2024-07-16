/**
 * Created by linpeng on 2024/7/4.
 * 计算同序列的最佳点
 */
#include "select.hpp"

// 辅助函数：计算两点间的距离
float distance2(const Point2f &p1, const Point2f &p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 辅助函数：判断点是否合理（基于预期距离）
bool isReasonablePoint(const LightPoint &p1, const LightPoint &p2, float avgDistance,
                       float tolerance = 0.5) {
    float expectedDist = abs(p1.label - p2.label) * avgDistance;
    float actualDist = distance2(p1.position, p2.position);
    return abs(actualDist - expectedDist) <= tolerance * expectedDist;
}

/**
 * 选择这个位置最适合的点
 */
LightPoint selectBestPoint(Mat &src, const vector<LightPoint> &totalPoints,
                           const vector<LightPoint> &samePoints,
                           vector<LightPoint> &errorPoints,
                           float avgDistance,
                           int neighborCount = 4) {
    if (samePoints.empty()) {
        throw runtime_error("samePoints is empty");
    }

    int targetSerial = samePoints[0].label;
    // 选择最佳点
    LightPoint bestPoint = samePoints[0];

    try {
        // 在 totalPoints 中找到目标序号的位置
        auto it = lower_bound(totalPoints.begin(), totalPoints.end(), targetSerial,
                              [](const LightPoint &point, int num) {
                                  return point.label < num;
                              });

        // 获取前后的多个点位
        vector<LightPoint> prevPoints, nextPoints;
        for (int i = 1; i <= neighborCount && it - i >= totalPoints.begin(); ++i) {
            prevPoints.push_back(*(it - i));
        }
        for (int i = 0; i < neighborCount && it + i != totalPoints.end(); ++i) {
            nextPoints.push_back(*(it + i));
        }


        float minDeviation = numeric_limits<float>::max();

        for (const auto &point: samePoints) {
            float totalDeviation = 0;
            int validNeighbors = 0;

            // 检查与前面点的关系
            for (const auto &prevPoint: prevPoints) {
                if (isReasonablePoint(point, prevPoint, avgDistance)) {
                    float expectedDist = abs(point.label - prevPoint.label) * avgDistance;
                    float actualDist = distance2(point.position, prevPoint.position);
                    totalDeviation += abs(actualDist - expectedDist) / expectedDist;
                    validNeighbors++;
                }
            }

            // 检查与后面点的关系
            for (const auto &nextPoint: nextPoints) {
                if (isReasonablePoint(point, nextPoint, avgDistance)) {
                    float expectedDist = abs(point.label - nextPoint.label) * avgDistance;
                    float actualDist = distance2(point.position, nextPoint.position);
                    totalDeviation += abs(actualDist - expectedDist) / expectedDist;
                    validNeighbors++;
                }
            }

            if (validNeighbors > 0) {
                float avgDeviation = totalDeviation / validNeighbors;
                if (avgDeviation < minDeviation) {
                    minDeviation = avgDeviation;
                    bestPoint = point;
                }
            }
        }

        // 将其他点添加到错误集合
        for (const auto &point: samePoints) {
            if (point.label != bestPoint.label || point.position != bestPoint.position) {
                errorPoints.push_back(point);
                LOGE(LOG_TAG, "---塞入异常点位 label =%d position = %f - %f", point.label,
                     point.position.x,
                     point.position.y);
                circle(src, point.position, 6, Scalar(0, 0, 255), 2);
                putText(src, to_string(point.label), point.position,
                        FONT_HERSHEY_SIMPLEX, 0.7,
                        Scalar(0, 0, 255), 1);
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "异常状态21");
    }

    return bestPoint;
}

/**
 * 处理得分一致的点，选出合适的点，其他的塞入错误点集合
 */
void
processSamePoints(Mat &src, vector<Mat> &outMats, vector<LightPoint> &totalPoints,
                  vector<LightPoint> &errorPoints,
                  float avgDistance, const map<int, vector<LightPoint>> sameSerialNumMap,int lightType) {
    sort(totalPoints.begin(), totalPoints.end(),
         [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

    Mat outMat = src.clone();
    for (const auto &entry: sameSerialNumMap) {
        vector<LightPoint> indices = entry.second;
        if (indices.size() > 1) {
            LightPoint bestPoint = selectBestPoint(outMat, totalPoints, indices, errorPoints,
                                                   avgDistance);
            LOGW(LOG_TAG, "---塞入正常点位 label =%d position = %f - %f", bestPoint.label,
                 bestPoint.position.x,
                 bestPoint.position.y);
            circle(outMat, bestPoint.position, 6, Scalar(0, 255, 0), 2);
            putText(outMat, to_string(bestPoint.label), bestPoint.position,
                    FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(0, 255, 0), 1);
            totalPoints.push_back(bestPoint);
        }
    }
    if (lightType != TYPE_H682X) {
        outMats.push_back(outMat);
    }
}
