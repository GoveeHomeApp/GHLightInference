#ifndef GroupUtilH61DX_HPP
#define GroupUtilH61DX_HPP

#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>
#include "UtilH61DX.hpp"

/// @brief 颜色分组
class GroupH61DX {
public:
    /// @brief 坐标点
    std::vector<cv::Point> points;

    /// @brief 颜色
    cv::Vec3b color;

    /// @brief 颜色对应的RGB值
    int rgb;

    GroupH61DX(cv::Vec3b color, std::vector<cv::Point> points) : color(color), points(points), rgb(H61DX::BGR2Int(color)) {};

    /// @brief 添加下一个分组
    void addNext(std::shared_ptr<GroupH61DX> next) {
        nexts.push_back(next);
    }

    /// @brief 获取所有下一个分组
    std::vector<std::shared_ptr<GroupH61DX>> getNexts() {
        std::vector<std::shared_ptr<GroupH61DX>> nexts;
        for (auto& weak : this->nexts) {
            auto next = weak.lock();
            if (next) {
                nexts.push_back(next);
            }
        }
        return nexts;
    }
    
#if DEBUG
    
    void debugPrint();

    char debugChar() {
        switch (rgb) {
            case H61DX::COLOR_RED:
                return 'R';
            case H61DX::COLOR_GREEN:
                return 'G';
            case H61DX::COLOR_BLUE:
                return 'B';
            case H61DX::COLOR_YELLOW:
                return 'Y';
            default:
                return 'X';
        }
    }
    
#endif

private:
    std::vector<std::weak_ptr<GroupH61DX>> nexts;
};


/// @brief 颜色分组工具
class GroupUtilH61DX
{
public:
    /// @brief 按相同颜色分组
    static std::vector<std::shared_ptr<GroupH61DX>> group(cv::Mat& image);
    /// @brief 获取最大跨度间距
    static int getSpan(cv::Mat& image, const cv::Point& start = cv::Point(-1, -1));
private:
    /// @brief 查找第一个非0的颜色
    static cv::Point findFirst(cv::Mat& image);
};

#endif // GroupUtilH61DX_HPP
