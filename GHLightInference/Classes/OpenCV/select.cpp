/**
 * Created by linpeng on 2024/7/4.
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
                LOGE(LOG_TAG, "---塞入异常点位 label =%d position = %d - %d", point.label,
                     point.position.x,
                     point.position.y);
                circle(src, point.position, 8, Scalar(0, 0, 255), 2);
                putText(src, to_string(point.label), point.position,
                        FONT_HERSHEY_SIMPLEX, 0.7,
                        Scalar(0, 0, 255), 2);
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "异常状态21");
    }

    return bestPoint;
}

void
processSamePoints(Mat &src, vector<Mat> &outMats, vector<LightPoint> &totalPoints,
                  vector<LightPoint> &errorPoints,
                  float avgDistance, map<int, vector<LightPoint>> sameSerialNumMap) {
    Mat outMat = src.clone();
    for (const auto &entry: sameSerialNumMap) {
        vector<LightPoint> indices = entry.second;
        if (indices.size() > 1) {
            LightPoint bestPoint = selectBestPoint(outMat, totalPoints, indices, errorPoints,
                                                   avgDistance);
            LOGW(LOG_TAG, "---塞入正常点位 label =%d position = %d - %d", bestPoint.label,
                 bestPoint.position.x,
                 bestPoint.position.y);
            circle(outMat, bestPoint.position, 8, Scalar(0, 255, 0), 2);
            putText(outMat, to_string(bestPoint.label), bestPoint.position,
                    FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(0, 255, 0), 2);
            totalPoints.push_back(bestPoint);
        }
    }
    outMats.push_back(outMat);
}
//vector<LightPoint>
//processSamePoints(Mat &src, vector<Mat> &outMats, LampBeadsProcessor &processor,
//                  double averageDistance) {
//    int sizeWithSame = normalPoints.size();
//    Mat outMat = src.clone();
//    vector<LightPoint> processedPoints;
//    auto findNearestLabels = [&](int currentLabel, int maxDiff = 20) {
//        vector<pair<int, Point2i>> nearLabels;
//        vector<pair<int, Point2i>> filteredLabels;
//        try {
//            for (const auto &entry: sameSerialNumMap) {
//                if (abs(entry.first - currentLabel) <= maxDiff && entry.first != currentLabel) {
//                    if (!entry.second.empty()) {
//                        nearLabels.emplace_back(entry.first, normalPoints[entry.second[0]].position);
//                    }
//                }
//            }
//
//            // 排序标签
//            sort(nearLabels.begin(), nearLabels.end(),
//                 [currentLabel](const auto &a, const auto &b) {
//                     return abs(a.first - currentLabel) < abs(b.first - currentLabel);
//                 });
//
//            // 改进的过滤逻辑
//            if (nearLabels.size() >= 2) {
//                for (size_t i = 0; i < nearLabels.size() - 1; ++i) {
//                    float distance = norm(nearLabels[i].second - nearLabels[i + 1].second);
//                    int labelDiff = abs(nearLabels[i].first - nearLabels[i + 1].first);
//                    float expectedDist = labelDiff * averageDistance;
//
//                    if (abs(distance - expectedDist) <= averageDistance) {
//                        if (filteredLabels.empty() ||
//                            filteredLabels.back().first != nearLabels[i].first) {
//                            filteredLabels.push_back(nearLabels[i]);
//                        }
//                        filteredLabels.push_back(nearLabels[i + 1]);
//                    }
//                }
//            }
//
//            // 如果过滤后没有点，至少保留最近的两个点
//            if (filteredLabels.empty() && nearLabels.size() >= 2) {
//                filteredLabels.push_back(nearLabels[0]);
//                filteredLabels.push_back(nearLabels[1]);
//            }
//        } catch (...) {
//            LOGE(LOG_TAG, "异常状态17");
//        }
//
//        return filteredLabels;
//    };
//
//    for (const auto &entry: sameSerialNumMap) {
//        int serialNum = entry.first;
//        const vector<int> &indices = entry.second;
//
//        if (indices.size() == 1) {
//            processedPoints.push_back(normalPoints[indices[0]]);
//        } else {
//            vector<LightPoint> candidatePoints;
//            for (int idx: indices) {
//                candidatePoints.push_back(normalPoints[idx]);
//            }
//
//            auto nearLabels = findNearestLabels(serialNum);
//            LOGD(LOG_TAG, "serialNum =%d  nearLabels = %d", serialNum, nearLabels.size());
//            LightPoint bestPoint = candidatePoints[0];
//            float minScore = std::numeric_limits<float>::max();
//
//            for (const auto &p: candidatePoints) {
//                float score = 0;
//                int comparedLabels = 0;
//
//                for (const auto &nearLabel: nearLabels) {
//                    if (comparedLabels >= 2) break;  // 只比较最近的两个标签
//                    float expectedDist = abs(nearLabel.first - serialNum) * averageDistance;
//                    float actualDist = norm(p.position - nearLabel.second);
//                    score += abs(actualDist - expectedDist);
//                    comparedLabels++;
//                }
//
//                if (score < minScore) {
//                    minScore = score;
//                    bestPoint = p;
//                }
//            }
//            LOGW(LOG_TAG, "正确点 label = %d  position = %d - %d", bestPoint.label,
//                 bestPoint.position.x, bestPoint.position.y);
//            circle(outMat, bestPoint.position, 8, Scalar(0, 255, 0), 2);
//            putText(outMat, to_string(bestPoint.label), bestPoint.position,
//                    FONT_HERSHEY_SIMPLEX, 0.7,
//                    Scalar(0, 255, 0), 2);
//            processedPoints.push_back(bestPoint);
//
//            for (const auto &p: candidatePoints) {
//                if (p.position.x != bestPoint.position.x || p.position.y != bestPoint.position.y) {
//                    LOGE(LOG_TAG, "错误点 label = %d   position = %d - %d ", p.label,
//                         p.position.x, p.position.y);
//                    circle(outMat, p.position, 8, Scalar(0, 0, 255), 2);
//                    putText(outMat, to_string(p.label), p.position,
//                            FONT_HERSHEY_SIMPLEX, 0.7,
//                            Scalar(0, 0, 255), 2);
//                    errorSerialVector.push_back(p);
//                }
//            }
//        }
//    }
//    // 这里可以根据需要处理abnormalPoints
//    outMats.push_back(outMat);
//    LOGW(LOG_TAG,
//         "delete same = %d  errorSerialVector = %d ",
//         processedPoints.size() - sizeWithSame, errorSerialVector.size());
//
//    return processedPoints;
//}
// 使用示例
//void processTotalPoints(vector<LightPoint>& totalPoints,
//                        const vector<LightPoint>& samePoints,
//                        vector<LightPoint>& errorPoints,
//                        float avgDistance) {
//    if (samePoints.empty()) return;
//
//    LightPoint bestPoint = selectBestPoint(totalPoints, samePoints, errorPoints, avgDistance);
//
//    // 在 totalPoints 中插入或更新最佳点
//    auto it = lower_bound(totalPoints.begin(), totalPoints.end(), bestPoint.label,
//                               [](const LightPoint& point, int num) {
//                                   return point.label < num;
//                               });
//
//    if (it != totalPoints.end() && it->label == bestPoint.label) {
//        *it = bestPoint;  // 更新现有点
//    } else {
//        totalPoints.insert(it, bestPoint);  // 插入新点
//    }
//}
