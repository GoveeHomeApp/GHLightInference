/**
 * Created by linpeng on 2024/7/4.
 *
对已知的矩形按序号排序。
遍历所有可能的序号，如果缺失，则进行插值。
插值使用线性插值方法，考虑了中心点、大小和角度
 */
#include "interpolate70cx.hpp"
#include <iostream>

std::vector<LightPoint>
completeLightPoints2D(const std::vector<LightPoint> &inputPoints, int maxNum) {
    std::vector<LightPoint> completedPoints;
    int oldSize = inputPoints.size();
    // 按lightIndex对输入点进行排序
    std::vector<LightPoint> sortedPoints = inputPoints;
    std::sort(sortedPoints.begin(), sortedPoints.end(),
              [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

    // 计算平均距离
    float avgDistance = 0;
    if (sortedPoints.size() > 1) {
        std::vector<float> distances;
        for (size_t i = 1; i < sortedPoints.size(); ++i) {
            cv::Point2f diff = sortedPoints[i].position - sortedPoints[i - 1].position;
            distances.push_back(cv::norm(diff));
        }
        avgDistance = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    }

    // 找到第一个和最后一个存在的索引
    int firstIndex = sortedPoints.front().label;
    int lastIndex = sortedPoints.back().label;

    // 如果必要，将范围扩展到0和maxNum
    firstIndex = std::min(firstIndex, 0);
    lastIndex = std::max(lastIndex, maxNum);

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
            auto prevIt = std::lower_bound(sortedPoints.begin(), sortedPoints.end(), currentIndex,
                                           [](const LightPoint &position, int index) {
                                               return position.label < index;
                                           }) - 1;
            auto nextIt = std::upper_bound(sortedPoints.begin(), sortedPoints.end(), currentIndex,
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
                cv::Point2f direction = prevIt->position - (prevIt - 1)->position;
                direction = direction / cv::norm(direction) * (avgDistance / 2);
                newPoint.position = prevIt->position + direction * (currentIndex - prevIt->label);
            } else if (nextIt < sortedPoints.end()) {
                // 在第一个已知点之前进行外推
                cv::Point2f direction = (nextIt + 1)->position - nextIt->position;
                direction = direction / cv::norm(direction) * (avgDistance / 2);
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
            maxGap = 2;
        } else {
            maxGap = 3;
        }

        if (gap > 0 && gap < maxGap) {
            for (int j = 1; j <= gap; ++j) {
                float t = static_cast<float>(j) / (gap + 1);
                Point2f newPoint = result[i].position * (1 - t) + result[i + 1].position * t;
                LightPoint lp = LightPoint();
                lp.label = result[i].label + j;
                lp.position = newPoint;
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
vector<GapInfo> analyzeGaps(const vector<Group> &groups, double threshold = 1.7) {
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

    return significantGaps;
}
