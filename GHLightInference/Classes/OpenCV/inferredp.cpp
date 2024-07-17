/**
 * Created by linpeng on 2024/7/4.
 *
对已知的矩形按序号排序。
遍历所有可能的序号，如果缺失，则进行插值。
插值使用线性插值方法，考虑了中心点、大小和角度
 */
#include "inferredp.hpp"
#include "sequence.hpp"

/**
 * 查找最接近的中心点
 * @param lA
 * @param lB
 * @param points
 * @param targetDistance
 * @return
 */
LightPoint findMostLikelyCenter(const LightPoint &lA, const LightPoint &lB, int inferredLightIndex,
                                int lightType,
                                unordered_map<int, vector<LightPoint>> &sequenceTypeMap,
                                double targetDistance1) {
    try {
        if (inferredLightIndex > getIcNum()) return LightPoint(EMPTY_POINT);

        int sequenceType = getNonSequenceType(inferredLightIndex, lightType);
        if (sequenceType == -1) {
            LOGE(TAG_INFERRED, "非推断序号");
            return LightPoint(EMPTY_POINT);
        }

        vector<LightPoint> points = sequenceTypeMap[sequenceType];
        Point2f A = lA.position;
        Point2f B = lB.position;

        double diffSegmentLenX = (A.x - B.x) / 2;
        double diffSegmentLenY = (A.y - B.y) / 2;
        double targetDistance = abs(distanceP(A, B));

        LightPoint mostLikelyC;
        float maxConfidence = std::numeric_limits<float>::lowest();
        float maxAllowedDistance = targetDistance * 0.6f;  // 最大允许距离

        Point2f AB = B - A;
        float AB_length = norm(AB);
        Point2f midpoint = A + 0.5 * AB;  // AB的中点
        int selectIndex = -1;
        for (int i = 0; i < points.size(); i++) {
            LightPoint lC = points[i];
            Point2f C = lC.position;
            float AC_distance = norm(C - A);
            float BC_distance = norm(C - B);

            // 如果AC或BC的距离超过最大允许距离，跳过这个点
            if (AC_distance > maxAllowedDistance || BC_distance > maxAllowedDistance) {
                continue;
            }

            // 检查AC和BC的距离是否接近目标距离
            float distanceDiff =
                    std::abs(AC_distance - targetDistance) + std::abs(BC_distance - targetDistance);

            // 计算C到AB中点的距离
            float distToMidpoint = norm(C - midpoint);

            // 计算C到AB线段的距离（用于判断C是否大致在AB的延长线上）
            float distToAB =
                    std::abs((C.y - A.y) * (B.x - A.x) - (C.x - A.x) * (B.y - A.y)) / AB_length;

            // 计算可信度（越大越好）
            float confidence = 1.0f / (distanceDiff + distToMidpoint + 2 * distToAB + 1e-5f);

            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                mostLikelyC = lC;
                selectIndex = i;
            }
        }

        // 如果没有找到符合条件的点，返回一个无效点
        if (maxConfidence == std::numeric_limits<float>::lowest()) {
            return LightPoint(EMPTY_POINT);
        }
        points.erase(points.begin() + selectIndex);
        sequenceTypeMap[sequenceType] = points;
        mostLikelyC.label = inferredLightIndex;
        return mostLikelyC;
    } catch (...) {
        LOGE(LOG_TAG, "findMostLikelyCenter error!");
        return LightPoint(EMPTY_POINT);
    }
}

/**
 * 寻找AB连线上的延伸点
 * @param A
 * @param B
 * @param targetDistance
 * @param points
 * @return
 */
LightPoint
findExtensionPointAB2C(const LightPoint &lA, const LightPoint &lB, int inferredLightIndex,
                       int lightType,
                       unordered_map<int, vector<LightPoint>> sequenceTypeMap,
                       double targetDistance) {
    try {
        if (inferredLightIndex > getIcNum()) return LightPoint(EMPTY_POINT);

        int sequenceType = getNonSequenceType(inferredLightIndex, lightType);
        if (sequenceType == -1) {
            LOGE(TAG_INFERRED, "非推断序号");
            return LightPoint(EMPTY_POINT);
        }
        vector<LightPoint> points = sequenceTypeMap[sequenceType];
        Point2f A = lA.position;
        Point2f B = lB.position;
        Point2f AB = B - A;
        Point2f AB_normalized = AB / norm(AB);

        LightPoint bestC;
        float bestScore = numeric_limits<float>::max();
        float maxAllowedDistance = targetDistance * 1.5f;

        int selectIndex = -1;
        for (int i = 0; i < points.size(); i++) {
            LightPoint lC = points[i];
            Point2f point = lC.position;
            // 计算点到B的距离
            float distanceToB = norm(point - B);

            // 如果距离超过允许的最大距离，跳过这个点
            if (distanceToB > maxAllowedDistance) {
                continue;
            }

            // 计算点到AB线的投影
            Point2f AP = point - A;
            float projection = AP.dot(AB_normalized);

            // 计算点到AB线的垂直距离
            Point2f projectionPoint = A + projection * AB_normalized;
            float perpendicularDistance = norm(point - projectionPoint);

            // 计算得分（越小越好）
            float score = abs(distanceToB - targetDistance) + perpendicularDistance;

            // 如果投影点在AB线段之外且在B点之后，则更优先考虑
            if (projection > norm(AB)) {
                score *= 0.5;
            }

            if (score < bestScore) {
                bestScore = score;
                bestC = lC;
                selectIndex = i;
            }
        }
        // 如果没有找到符合条件的点，返回一个无效点
        if (selectIndex == -1) {
            return LightPoint(EMPTY_POINT);
        }
        points.erase(points.begin() + selectIndex);
        bestC.label = inferredLightIndex;
        return bestC;
    } catch (...) {
        LOGE(LOG_TAG, "findMostLikelyCenter error!");
        return LightPoint(EMPTY_POINT);
    }
}

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
