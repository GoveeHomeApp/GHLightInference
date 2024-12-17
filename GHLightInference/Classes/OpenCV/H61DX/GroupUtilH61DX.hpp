#ifndef GroupUtilH61DX_HPP
#define GroupUtilH61DX_HPP

#include <vector>
#include <functional>
#include <optional>
#include <opencv2/core/mat.hpp>
#include "UtilH61DX.hpp"
#include <memory>

/// @brief 颜色分组
class GroupH61DX: public std::enable_shared_from_this<GroupH61DX> {
public:
    /// @brief 坐标点
    std::vector<cv::Point> points;

    /// @brief 颜色
    cv::Vec3b color;

    /// @brief 颜色对应的RGB值
    int rgb;
    
    /// 分块大小
    int blockWidth;
    
    /// @brief 分组被分割后的块数
    std::vector<cv::Point> blockCenters;

    /// @brief 所有块数的平均中点
    cv::Point allBlocksAvgCenter;

    GroupH61DX(cv::Vec3b color, std::vector<cv::Point> points, int span) :
    color(color),
    points(points),
    rgb(H61DX::BGR2Int(color)),
    blockWidth(span)
    {
        this->slicingBlock();
    };
    
    std::shared_ptr<GroupH61DX> getSharedPtr() {
        return shared_from_this();
    }


    /// @brief 添加下一个分组
    void addNext(std::shared_ptr<GroupH61DX> next) {
        nexts.push_back(next);
    }

    /// @brief 检查并添加下一个分组
    void checkAddNext(std::shared_ptr<GroupH61DX> next) {
        for (auto& weak : nexts) {
            if (weak.lock() == next) {
                return;
            }
        }
        addNext(next);
    }

    /// @brief 移除一个分组
    void removeNext(std::shared_ptr<GroupH61DX> next) {
        for (auto it = nexts.begin(); it != nexts.end(); it++) {
            if (it->lock() == next) {
                nexts.erase(it);
                break;
            }
        }
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

    /// 获取距离指定分组中最近的块中心
    cv::Point getClosestBlockCenter(const std::shared_ptr<GroupH61DX>& group);

    /// 获取连接路径中心点（从from分组开始，到to分组结束，默认为空）
    std::vector<cv::Point> getPathCenters(const std::shared_ptr<GroupH61DX>& fromGroup = nullptr, const std::shared_ptr<GroupH61DX>& toGroup = nullptr, int span = 3);
    
    /// 计算与group直接的距离
    int distanceWithGroup(std::shared_ptr<GroupH61DX> group, bool useBlockCenter = true);
    
    /// 将other合并到当前分组
    void merge(std::shared_ptr<GroupH61DX> other);

    /// 根据选择下一个分组
    std::vector<std::shared_ptr<GroupH61DX>> pickNexts(int rgb, std::shared_ptr<GroupH61DX> fromGroup = nullptr);
    
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
    
    /// 对坐标进行分块
    void slicingBlock();
};


/// @brief 颜色分组工具
class GroupUtilH61DX
{
public:
    /// @brief 按相同颜色分组
    static std::vector<std::shared_ptr<GroupH61DX>> group(cv::Mat& image, int span);
    /// @brief 获取最大跨度间距
    static int getSpan(cv::Mat& image, const cv::Point& start = cv::Point(-1, -1));
    /// @brief 查找第一个非0的颜色
    static cv::Point findFirst(cv::Mat& image);
};

#endif // GroupUtilH61DX_HPP
