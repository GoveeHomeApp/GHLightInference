#include "visualization.hpp"

void Visualizer::visualizeSplitState(
        const Mat &image,
        const vector<PointSet> &sets,
        const vector<LedPoint> &confirmed_points,
        int split_round,
        vector<Mat> &outMats
) {
    if (!Logger::debugSwitch)return;
    Mat visualMat = image.clone();

    // 绘制已确认的点
    drawConfirmedPoints(visualMat, confirmed_points);

    // 绘制未确认的点集
    drawPointSets(visualMat, sets);

    // 添加分割轮次标题
    putText(visualMat, "Split Round: " + to_string(split_round),
            Point(50, 50), FONT_HERSHEY_SIMPLEX,
            0.7, Scalar(255, 0, 50), 2);

    outMats.push_back(visualMat);
}

void Visualizer::drawConfirmedPoints(Mat &image, const vector<LedPoint> &points) {
    for (const auto &point: points) {
        // 使用白色圆圈标记已确认的点
        circle(image, point.position, 8, Scalar(255, 255, 255), 2);
        // 显示点的ID
        putText(image, to_string(point.id),
                point.position,
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(255, 255, 255),
                1);
    }
}

void Visualizer::drawPointSets(Mat &image, const vector<PointSet> &sets) {
    for (const auto &set: sets) {
        // 为每个集合中的点选择颜色
        Scalar color;
        if (!set.points.empty()) {
            // 根据第一个点的颜色属性决定整个集合的颜色
            Vec3b avg_color = set.points[0].color;
            if (set.colorType == 0) { // 如果红色分量大于绿色分量
                color = Scalar(0, 0, 255); // 红色
            } else {
                color = Scalar(0, 255, 0); // 绿色
            }
        }

        // 绘制该集合中的所有点
        for (const auto &point: set.points) {
            if (!point.confirmed) {  // 只绘制未确认的点
                circle(image, point.position, 8, color, 2);
                // 显示点所属的范围
                string range_text = to_string(set.start_range) + "-" +
                                    to_string(set.end_range);
                putText(image, range_text,
                        Point(point.position.x + 10, point.position.y),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1);
            }
        }
    }
} 