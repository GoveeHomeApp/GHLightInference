/**
 * Created by linpeng on 2024/7/4.
 *
对已知的矩形按序号排序。
遍历所有可能的序号，如果缺失，则进行插值。
插值使用线性插值方法，考虑了中心点、大小和角度
 */
#include "inferredp.hpp"
#include <iostream>

Point2f findMostLikelyCenter(const Point2f &A, const Point2f &B,
                             const std::vector<Point2f> &points, float targetDistance) {
    Point2f mostLikelyC;
    float maxConfidence = std::numeric_limits<float>::lowest();
    float maxAllowedDistance = targetDistance * 1.5f;  // 最大允许距离

    Point2f AB = B - A;
    float AB_length = norm(AB);
    Point2f midpoint = A + 0.5 * AB;  // AB的中点

    for (const auto &C: points) {
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
            mostLikelyC = C;
        }
    }

    // 如果没有找到符合条件的点，返回一个无效点
    if (maxConfidence == std::numeric_limits<float>::lowest()) {
        return Point2f(std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN());
    }

    return mostLikelyC;
}

Point2f findPointC(const Point2f &A, const Point2f &B, float targetDistance,
                                  const std::vector<Point2f> &points) {
    Point2f AB = B - A;
    Point2f AB_normalized = AB / norm(AB);

    Point2f bestC;
    float bestScore = std::numeric_limits<float>::max();
    float maxAllowedDistance = targetDistance * 1.5f;

    for (const auto &point: points) {
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
        float score = std::abs(distanceToB - targetDistance) + perpendicularDistance;

        // 如果投影点在AB线段之外且在B点之后，则更优先考虑
        if (projection > norm(AB)) {
            score *= 0.5;
        }

        if (score < bestScore) {
            bestScore = score;
            bestC = point;
        }
    }

    return bestC;
}
