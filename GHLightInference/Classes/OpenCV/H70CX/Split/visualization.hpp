#pragma once
#include "point_types.hpp"
#include <opencv2/opencv.hpp>

class Visualizer {
public:
    // 可视化当前分割状态
    static void visualizeSplitState(
        const cv::Mat& image,
        const std::vector<PointSet>& sets,
        const std::vector<LedPoint>& confirmed_points,
        int split_round,
        std::vector<cv::Mat>& outMats
    );
    
private:
    // 绘制确认点位
    static void drawConfirmedPoints(
        cv::Mat& image,
        const std::vector<LedPoint>& points
    );
    
    // 绘制未确认的点位集合
    static void drawPointSets(
        cv::Mat& image,
        const std::vector<PointSet>& sets
    );
}; 