#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <stdexcept>
#include "LogUtils.h"

using namespace cv;
using namespace std;

// 自定义异常类
class PointException : public runtime_error {
public:
    using runtime_error::runtime_error;
};

// LED点位信息结构体
struct LedPoint {
    int id;                 // 点位序号(0-1000)
    Point2f position;   // 点位坐标
    Vec3b color;        // 点位颜色
    bool confirmed;         // 序号是否已确认
    float radius;          // 灯珠半径
    LedPoint(int _id = -1, Point2f _pos = Point2f(0, 0), float r = 7.0f)
            : id(_id), position(_pos), confirmed(false), radius(r) {
        validate();
    }

    // 验证点位数据
    void validate() const {
        if (id > 1000 || id < -1) {
            throw PointException("Invalid point ID: " + to_string(id));
        }
        if (position.x < 0 || position.y < 0) {
            throw PointException("Invalid position");
        }
        if (radius <= 0) {
            throw PointException("Invalid radius: " + to_string(radius));
        }
    }

    // 计算点位的平均颜色
    Vec3b calculateAverageColor(const Mat &image) const;
};

// 点位集合类
class PointSet {
public:
    vector<LedPoint> points;    // 点位集合
    int start_range;                 // 序号范围起始值
    int end_range;                   // 序号范围结束值
    int colorType;                   // 0 红色，1 绿色

    PointSet(int start = 0, int end = 1000)
            : start_range(start), end_range(end) {
        validateRange();
    }

    // 验证范围有效性
    void validateRange() const;

    // 验证点位集合
    bool validatePoints() const;

    // 计算两点之间的距离
    static float distance(const LedPoint &p1, const LedPoint &p2);

    // 判断是否为线性均匀分布
    bool isLinearDistribution() const;

    // 获取点位数量
    size_t size() const { return points.size(); }

    int range() const { return end_range - start_range + 1; }
};
