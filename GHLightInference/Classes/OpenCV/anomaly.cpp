#include "anomaly.hpp"

/**
 * Created by linpeng on 2024/7/3.
 * 丢弃偏差序号点
 */
class AnomalyDetector {
private:
    vector<LightPoint> points;  // 所有点的信息
    float avgDistance;         // 平均点间距离
    float distanceThreshold;   // 距离阈值系数
    int totalPoints;           // 总点数
    int maxLabel;              // 最大标签值
    int expectedNeighbors;     // 期望的邻近点数量
    int densityNeighbors;  // 用于计算密度的邻居数
    int baseLabelDiffThreshold = 15;  // 基础标签差阈值

    // 构建KD树用于快速邻近搜索
    void buildKDTree(vector<vector<pair<float, int>>> &kdTree) {
        kdTree.resize(totalPoints);
        for (int i = 0; i < totalPoints; ++i) {
            for (int j = 0; j < totalPoints; ++j) {
                if (i != j) {
                    float dist = norm(points[i].position - points[j].position);
                    if (dist <= avgDistance * distanceThreshold) {
                        kdTree[i].emplace_back(dist, j);
                    }
                }
            }
            sort(kdTree[i].begin(), kdTree[i].end());  // 按距离排序
        }
    }

    void calculateLocalDensity(const vector<vector<pair<float, int>>> &kdTree) {
        for (int i = 0; i < totalPoints; ++i) {
            int count = min(densityNeighbors, static_cast<int>(kdTree[i].size()));
            if (count > 0) {
                try {
                    vector<float> neighborLightIndices;
                    for (int j = 0; j < count; ++j) {
                        if (kdTree[i][j].second != i) {  // 排除自身
                            neighborLightIndices.push_back(points[kdTree[i][j].second].label);
                        }
                    }

                    if (!neighborLightIndices.empty()) {
                        // 计算邻居lightIndex的中位数
                        size_t n = neighborLightIndices.size() / 2;
                        nth_element(neighborLightIndices.begin(), neighborLightIndices.begin() + n,
                                    neighborLightIndices.end());
                        float medianLightIndex = neighborLightIndices[n];

                        // 如果邻居数量为偶数，取中间两个数的平均值
                        if (neighborLightIndices.size() % 2 == 0 &&
                            neighborLightIndices.size() > 1) {
                            auto max_it = max_element(neighborLightIndices.begin(),
                                                      neighborLightIndices.begin() + n);
                            medianLightIndex = (medianLightIndex + *max_it) / 2.0f;
                        }

                        // 计算当前点lightIndex与邻居中位数的绝对差距
                        points[i].localDensity = abs(points[i].label - medianLightIndex);
                    } else {
                        points[i].localDensity = baseLabelDiffThreshold * 2;  // 如果只有自己，设置一个较大的值
                    }
                } catch (...) {
                    points[i].localDensity = baseLabelDiffThreshold;
                }
            } else {
                points[i].localDensity = baseLabelDiffThreshold;  // 如果没有邻居，设置一个较大的值
            }
        }
    }

    // 为指定范围的点找邻近点
    void findNeighborsForRange(int start, int end, const vector<vector<pair<float, int>>> &kdTree) {
        for (int i = start; i < end; ++i) {
            int adjustedThreshold = max(baseLabelDiffThreshold,
                                        min(static_cast<int>(ceil(points[i].localDensity)),
                                            maxLabel / 10));
            for (const auto &pair: kdTree[i]) {
                double dist = pair.first;
                int j = pair.second;
                // 使用dist和j
                if (abs(points[i].label - points[j].label) <= adjustedThreshold) {
                    points[i].neighbors.push_back(j);
                }
            }
        }
    }

    // 并行处理找邻近点
    void findNeighbors() {
        vector<vector<pair<float, int>>> kdTree;
        buildKDTree(kdTree);
        calculateLocalDensity(kdTree);

        unsigned int numThreads = thread::hardware_concurrency();
        numThreads =
                numThreads > 0 ? min(numThreads, static_cast<unsigned int>(totalPoints / 100 + 1))
                               : 2;

        vector<future<void>> futures;
        int chunkSize = totalPoints / numThreads;

        for (unsigned int i = 0; i < numThreads; ++i) {
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? totalPoints : (i + 1) * chunkSize;
            futures.push_back(
                    async(launch::async, &AnomalyDetector::findNeighborsForRange, this, start, end,
                          ref(kdTree)));
        }

        for (auto &f: futures) {
            f.wait();
        }
    }

    // 判断一个点是否为异常点
    bool isAnomaly(int index) {
        const auto &point = points[index];
//        if (point.label == 236 || point.label == 350) {
//            LOGD(TAG_DELETE, "--isAnomaly----- -neighbors = %d ", point.neighbors.size());
//            if (point.neighbors.size() > 0) {
//                LOGD(TAG_DELETE, "--isAnomaly----- -neighbors label= %d   localDensity = %d",
//                     points[point.neighbors[0]].label,
//                     point.localDensity);
//            }
//        }
        if (point.neighbors.empty()) return true;  // 没有邻近点，视为异常

        // 检查邻近点数量是否符合预期
        int actualExpectedNeighbors = min(expectedNeighbors, maxLabel - point.label);
        if (point.neighbors.size() < actualExpectedNeighbors / 2) {
            LOGW(TAG_DELETE, " neighbors = %d  actualExpectedNeighbors=%d", point.neighbors.size(),
                 actualExpectedNeighbors / 2);
            return true;
        }

        // 检查邻近点的标签一致性
        int consistentNeighbors = 0;
        for (int neighborIdx: point.neighbors) {
            if (points.size() <= neighborIdx) {
                LOGE(TAG_DELETE, "---------异常状态15");
                continue;
            }
            int neighborOffset = abs(points[neighborIdx].label - point.label);
//            if (point.label == 236 || point.label == 350) {
//                LOGD(TAG_DELETE,
//                     "--------label = %d -neighborOffset = %d  baseLabelDiffThreshold = %d",
//                     point.label, neighborOffset, baseLabelDiffThreshold);
//            }
            if (neighborOffset < baseLabelDiffThreshold * 2) {
                consistentNeighbors++;
            }
        }

        return consistentNeighbors < 1;
    }

public:
    // 构造函数
    AnomalyDetector(const vector<LightPoint> &pts, float avgDist, float distThresh,
                    int baseLabelDiffThresh, int maxLabelValue, int densityNeighborCount,
                    int expectedNeighborCount = 2)
            : points(pts), avgDistance(avgDist), distanceThreshold(distThresh),
              baseLabelDiffThreshold(baseLabelDiffThresh), totalPoints(pts.size()),
              maxLabel(maxLabelValue), expectedNeighbors(expectedNeighborCount),
              densityNeighbors(densityNeighborCount) {
        findNeighbors();
    }

    // 检测并返回所有异常点的索引
    vector<int> detectAnomalies() {
        vector<int> anomalies;
        for (int i = 0; i < totalPoints; ++i) {
            if (isAnomaly(i)) {
                anomalies.push_back(i);
            }
        }
        return anomalies;
    }
};

//int main() {
//    // 假设我们已经有了点的位置和标签
//    vector<LightPoint> points;
//    // 初始化点的信息...
//
//    float avgDistance = 10.0f;    // 假设的平均距离
//    float distanceThreshold = 2.5f;  // 距离阈值系数
//    int labelDiffThreshold = 4;   // 允许的最大标签差
//    int maxLabel = 1000;          // 最大标签值
//    int expectedNeighbors = 8;    // 期望的邻近点数量
//
//    // 创建异常检测器实例
//    AnomalyDetector detector(points, avgDistance, distanceThreshold,
//                             labelDiffThreshold, maxLabel, expectedNeighbors);
//
//    // 执行异常检测
//    vector<int> anomalies = detector.detectAnomalies();
//
//    // 输出异常点
//    for (int anomalyIndex : anomalies) {
//        cout << "Point " << points[anomalyIndex].label << " is anomalous." << endl;
//    }
//
//    return 0;
//}
