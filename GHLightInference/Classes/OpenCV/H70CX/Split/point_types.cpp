#include "point_types.hpp"

// 计算点位的平均颜色
Vec3b LedPoint::calculateAverageColor(const Mat& image) const {
    Vec3d color_sum(0, 0, 0);
    int pixel_count = 0;

    // 计算圆形区域的边界框
    int x_start = max(0, static_cast<int>(position.x - radius));
    int x_end = min(image.cols - 1, static_cast<int>(position.x + radius));
    int y_start = max(0, static_cast<int>(position.y - radius));
    int y_end = min(image.rows - 1, static_cast<int>(position.y + radius));

    // 遍历边界框内的像素
    for (int y = y_start; y <= y_end; y++) {
        for (int x = x_start; x <= x_end; x++) {
            // 检查像素是否在圆内
            float dx = x - position.x;
            float dy = y - position.y;
            if (dx*dx + dy*dy <= radius*radius) {
                Vec3b pixel = image.at<Vec3b>(y, x);
                color_sum[0] += pixel[0];
                color_sum[1] += pixel[1];
                color_sum[2] += pixel[2];
                pixel_count++;
            }
        }
    }

    // 计算平均值
    if (pixel_count > 0) {
        return Vec3b(
            static_cast<uchar>(color_sum[0] / pixel_count),
            static_cast<uchar>(color_sum[1] / pixel_count),
            static_cast<uchar>(color_sum[2] / pixel_count)
        );
    }

    return {0, 0, 0};
}

// 验证范围有效性
void PointSet::validateRange() const {
    if (start_range < 0 || end_range > 1000 || start_range > end_range) {
        throw PointException("Invalid range: [" +
            to_string(start_range) + "," +
            to_string(end_range) + "]");
    }
}

// 验证点位集合
bool PointSet::validatePoints() const {
    for (const auto& point : points) {
        if (point.id != -1 && (point.id < start_range || point.id > end_range)) {
            return false;
        }
    }
    return true;
}

// 计算两点之间的距离
float PointSet::distance(const LedPoint& p1, const LedPoint& p2) {
    float dx = p1.position.x - p2.position.x;
    float dy = p1.position.y - p2.position.y;
    return sqrt(dx * dx + dy * dy);
}

// 判断是否为线性均匀分布
bool PointSet::isLinearDistribution() const {
    if (points.size() < 3) return false;

    LOGD(LOG_TAG, "开始线性分布检测 - 点数: %zu", points.size());

    try {
        // 创建单通道输入矩阵 (N x 2 矩阵，每行是一个点的x,y坐标)
        Mat_<float> data(points.size(), 2);
        for (size_t i = 0; i < points.size(); i++) {
            data(i, 0) = points[i].position.x;
            data(i, 1) = points[i].position.y;
            LOGD(LOG_TAG, "点 %zu: (%.2f, %.2f)", i, points[i].position.x, points[i].position.y);
        }

        // 计算均值
        Mat mean;
        reduce(data, mean, 0, REDUCE_AVG);

        // 中心化数据
        Mat centered = data - repeat(mean, data.rows, 1);

        // 计算协方差矩阵
        Mat covar, eigenvectors, eigenvalues;
        calcCovarMatrix(centered, covar, mean, COVAR_NORMAL | COVAR_ROWS);

        // 计算特征值和特征向量
        eigen(covar, eigenvalues, eigenvectors);

        // 获取主方向（第一个特征向量）
        Point2f principal_vector(eigenvectors.at<float>(0,0),
                                   eigenvectors.at<float>(0,1));

        LOGD(LOG_TAG, "主方向向量: (%.2f, %.2f)",
             principal_vector.x, principal_vector.y);

        // 计算点到直线的最大偏差
        float max_deviation = 0;
        for (const auto& point : points) {
            Point2f vec(point.position.x - mean.at<float>(0,0),
                          point.position.y - mean.at<float>(0,1));
            float deviation = abs(vec.x * principal_vector.y -
                                     vec.y * principal_vector.x);
            max_deviation = max(max_deviation, deviation);
        }

        // 计算点之间的平均间距
        float total_distance = 0;
        vector<float> distances;
        for (size_t i = 1; i < points.size(); i++) {
            float dist = distance(points[i], points[i-1]);
            distances.push_back(dist);
            total_distance += dist;
        }
        float avg_distance = total_distance / distances.size();

        // 计算间距的标准差
        float variance = 0;
        for (float dist : distances) {
            float diff = dist - avg_distance;
            variance += diff * diff;
        }
        variance /= distances.size();
        float std_dev = sqrt(variance);

        LOGD(LOG_TAG, "线性分布检测指标:");
        LOGD(LOG_TAG, "- 最大偏差: %.2f", max_deviation);
        LOGD(LOG_TAG, "- 平均间距: %.2f", avg_distance);
        LOGD(LOG_TAG, "- 标准差: %.2f", std_dev);

        // 判断条件：
        // 1. 最大偏差不超过平均间距的20%
        // 2. 间距的标准差不超过平均间距的15%
        // 3. 点数量与范围大小一致
        bool is_linear = max_deviation < (avg_distance * 0.2) &&
                        std_dev < (avg_distance * 0.15) &&
                        points.size() == (end_range - start_range + 1);

        LOGD(LOG_TAG, "线性分布检测结果: %s", is_linear ? "是" : "否");
        LOGD(LOG_TAG, "- 偏差比例: %.2f%%", (max_deviation / avg_distance) * 100);
        LOGD(LOG_TAG, "- 标准差比例: %.2f%%", (std_dev / avg_distance) * 100);

        return is_linear;

    } catch (...) {
//        LOGE(LOG_TAG, "PCA计算错误: %s", e.what());
        return false;
    }
} 
