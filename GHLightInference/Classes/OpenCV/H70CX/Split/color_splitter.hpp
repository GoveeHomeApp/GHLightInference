#pragma once

#include "point_types.hpp"
#include "visualization.hpp"
#include <map>
#include <memory>
#include "LogUtils.h"

using namespace cv;
using namespace std;

// 分割器配置结构体
struct SplitterConfig {
    int min_set_size = 8;                    // 最小集合大小阈值，当点集小于此值时尝试推导序号
    float linear_deviation_threshold = 0.2f;  // 线性分布的最大偏差阈值
    float distance_std_threshold = 0.15f;     // 点间距标准差阈值
    int max_id_range = 1000;                 // 最大序号范围（0到max_id_range）

    void validate() const {
        if (min_set_size < 2) {
            throw PointException("Invalid min_set_size: must be >= 2");
        }
        if (linear_deviation_threshold <= 0 || linear_deviation_threshold >= 1) {
            throw PointException("Invalid linear_deviation_threshold: must be between 0 and 1");
        }
        if (distance_std_threshold <= 0 || distance_std_threshold >= 1) {
            throw PointException("Invalid distance_std_threshold: must be between 0 and 1");
        }
        if (max_id_range <= 0) {
            throw PointException("Invalid max_id_range: must be > 0");
        }
    }
};

// 分割策略接口 - 允许扩展不同的分割算法
class SplitStrategy {
public:
    virtual vector<PointSet> split(
            const PointSet &input,
            const Mat &image
    ) = 0;

    int split_count;  // 分割计数器
    virtual ~SplitStrategy() = default;
};

// 基于颜色的分割策略实现
class ColorBasedSplitStrategy : public SplitStrategy {
public:
    int color_channels_ = 2;

    vector<PointSet> split(
            const PointSet &input,
            const Mat &image
    ) override;

};

class ColorSplitter {
public:
    // 颜色枚举
    enum Color {
        BLACK = -16711423,  // 已确认点位的颜色
        RED = -65536,
        GREEN = -16711936
//        BLUE = -16776961
    };

    // 构造函数接受配置对象
    explicit ColorSplitter(const SplitterConfig &config = SplitterConfig());

    // 带初始点位的构造函数
    ColorSplitter(const vector<LedPoint> &initial_points,
                  const SplitterConfig &config = SplitterConfig());

    // 处理新的图片输入，返回是否所有点位都已确认
    bool processImage(const Mat &image, vector<Mat> &outMats);

    bool initialize(const Mat &image, vector<Point2f> pPointXys, vector<Mat> &outMats);

    // 获取当前已确认的点位
    vector<LedPoint> getConfirmedPoints() const { return confirmed_points_; }

    // 获取当前未确认的点位集合
    const vector<PointSet> &getPointSets() const { return point_sets_; }

    // 检查是否所有点位都已确认
    bool isAllPointsConfirmed() const;

    // 获取最大需要的分割次数
    int initMaxSplitCount() {
        int result = config_.max_id_range;
        while (result > 2) {
            result = result / color_channels_;
            LOGV("getLightColors", "getMaxSplitCount result=%d", result);
            maxSplitCount++;
        }
        return maxSplitCount;
    }

    int getMaxSplitCount() const {
        return maxSplitCount;
    }

    int getIcNum() const {
        return config_.max_id_range + 1;
    }

    // 获取指定分割次数的序号颜色映射
    // 返回值：vector<vector<int>>
    // - 第一个vector包含所有序号和对应的颜色值对
    // - 后续三个vector分别包含RED、GREEN、BLUE组的序号
    vector<vector<int>> getSplitColorMapping(int split_level) const;

    // 设置分割策略
    void setStrategy(shared_ptr<SplitStrategy> strategy) {
        strategy_ = strategy;
    }

    // 清除缓存
    void clearCache() {
        point_sets_.clear();
        confirmed_points_.clear();
        color_mapping_cache_.clear();
    }

    void initColorMappingCache();

    void calculateRangeMappings(
            int start_range,
            int end_range,
            int target_level,
            int current_level,
            vector<vector<int>> &level_mapping
    );

private:
    int maxSplitCount = 0;
    SplitterConfig config_;                  // 配置参数
    shared_ptr<SplitStrategy> strategy_;// 分割策略
    mutable map<int, vector<vector<int>>>
            color_mapping_cache_; // 颜色映射缓存
    int color_channels_ = 2;
    vector<PointSet> point_sets_;       // 当前的点位集合
    vector<LedPoint> confirmed_points_;    // 已确认序号的点位

    // 处理小规模集合
    void processSmallSet(PointSet &set);

    // 根据相邻关系推导序号
    void deducePointIds(PointSet &set);

    // 随机分配序号(当无法推导时使用)
    void assignRandomIds(PointSet &set);

    // 分配ID并确认点位
    void assignIds(PointSet &set, int first_index, int second_index);

    // 选择最优点（当点数超过范围时）
    void selectOptimalPoints(PointSet &set, const Mat &image, float distanceAvg);
}; 
