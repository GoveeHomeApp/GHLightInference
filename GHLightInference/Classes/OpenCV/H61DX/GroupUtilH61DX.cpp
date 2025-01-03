#include "GroupUtilH61DX.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "UtilH61DX.hpp"
#include <unordered_map>
#include <unordered_set>
#include "LogUtils.h"

#if DEBUG
#include <iostream>
#endif

using namespace cv;
using namespace std;
using namespace H61DX;

namespace
{
    typedef GroupH61DX Group;

    /// 获取当前点周围一圈的点的位置（仅获取指定跨度的那一圈）
    vector<Point> getRangeLocations(const Point &now, int span)
    {
        if (span == 0)
        {
            return {};
        }
        vector<Point> result;
        int x = now.x, y = now.y, s = span;

        for (int a = -s + 1; a <= s; ++a)
        {
            result.push_back(Point(x + a, y - s)); // 顶部
        }
        for (int a = -s + 1; a <= s; ++a)
        {
            result.push_back(Point(x + s, y + a)); // 右侧
        }
        for (int a = -s; a <= s - 1; ++a)
        {
            result.push_back(Point(x + a, y + s)); // 底部
        }
        for (int a = -s; a <= s - 1; ++a)
        {
            result.push_back(Point(x - s, y + a)); // 左侧
        }

        return result;
    }

    /// 获取当前点到指定跨度内的所有点，包含的所有圈，如span=3，则会获取当前点1、2、3三圈的所有点，从内至外
    vector<Point> getRangeAllLocations(const Point &now, int span, bool fromInside = true)
    {
        if (span < 1)
        {
            return {};
        }
        vector<Point> result;

        for (int s = 1; s <= span; ++s)
        {
            if (fromInside)
            {
                vector<Point> currentCircle = getRangeLocations(now, s);
                result.insert(result.end(), currentCircle.begin(), currentCircle.end());
            }
            else
            {
                vector<Point> currentCircle = getRangeLocations(now, s);
                result.insert(result.begin(), currentCircle.begin(), currentCircle.end());
            }
        }

        return result;
    }

    /// 获取对应像素点的颜色
    cv::Vec3b getColor(const cv::Mat &image, const cv::Point &point)
    {
        return image.at<cv::Vec3b>(point);
    }

    /// 获取对应像素点的颜色RGB值
    int getColorRGB(const cv::Mat &image, const cv::Point &point)
    {
        return BGR2Int(getColor(image, point));
    }

    /// 检查是否已经被访问
    bool isVisited(const vector<vector<bool>> &visited, const cv::Point &point)
    {
        return visited[point.y][point.x];
    }

    void setVisited(vector<vector<bool>> &visited, const cv::Point &point)
    {
        visited[point.y][point.x] = true;
    }

    /// 检查是否已经被访问，并将其标记为已访问
    bool checkAndSetVisited(vector<vector<bool>> &visited, const cv::Point &point)
    {
        if (isVisited(visited, point))
        {
            return true;
        }
        setVisited(visited, point);
        return false;
    }

    /// 获取中位数
    int getMedian(vector<int> &values)
    {
        if (values.empty())
        {
            return 0;
        }
        sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    /// 获取所有点的平均值
    Point getAvgPoint(vector<Point> &values)
    {
        if (values.empty())
        {
            return Point(0, 0);
        }
        int sumX = 0, sumY = 0;
        for (auto &point : values)
        {
            sumX += point.x;
            sumY += point.y;
        }
        return Point(sumX / values.size(), sumY / values.size());
    }

    /// 求向量AB和CD之间的夹角（取劣角）
    double getLinesAngle(Point A, Point B, Point C, Point D)
    {
        auto ab = atan2(B.y - A.y, B.x - A.x);
        auto cd = atan2(D.y - C.y, D.x - C.x);
        auto result = abs(ab - cd);
        if (result > CV_2PI) {
            return result - CV_PI;
        } else if (result > CV_PI) {
            return CV_2PI - result;
        }
        return result;
    }

    /// 求向量BC和AB之间的夹角（取劣角）
    double getTargetPointAngle(Point A, Point B, Point C)
    {
        return getLinesAngle(A, B, B, C);
    }

    /// 计算两点之间的距离，八个方向
    int distancePoint(Point a, Point b)
    {
        return max(abs(a.x - b.x), abs(a.y - b.y));
    }

    /// 计算两个分组之间的距离，取最小距离（以所有坐标来计算）
    int distanceGroup(std::shared_ptr<Group> a, std::shared_ptr<Group> b)
    {
        int minDistance = INT_MAX;
        for (auto &pointA : a->points)
        {
            for (auto &pointB : b->points)
            {
                int distance = distancePoint(pointA, pointB);
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }
        return minDistance;
    }

    /// 计算两个分组之间的距离，取最小距离（以边框来计算）
    int distanceBorderGroup(std::shared_ptr<Group> a, std::shared_ptr<Group> b)
    {
        int minDistance = INT_MAX;
        for (auto &pointA : a->borderPoints)
        {
            for (auto &pointB : b->borderPoints)
            {
                int distance = distancePoint(pointA, pointB);
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }
        return minDistance;
    }

    /// 计算两个分组之间的距离，取最小距离（以所分块中心来计算）
    int blockCentersDistance(std::shared_ptr<Group> a, std::shared_ptr<Group> b)
    {
        int minDistance = INT_MAX;
        for (auto &pointA : a->blockCenters)
        {
            for (auto &pointB : b->blockCenters)
            {
                int distance = distancePoint(pointA, pointB);
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }
        return minDistance;
    }

    /// 合并两个分组(b合并到a中)
    void mergeGroup(std::shared_ptr<Group> a, std::shared_ptr<Group> b)
    {
        auto medianArea = static_cast<double>(a->points.size())/max(double(1), a->areaRate);
        a->points.insert(a->points.end(), b->points.begin(), b->points.end());
        a->updateMaxAcrossCount(static_cast<int>(medianArea));
        a->blockCenters.insert(a->blockCenters.end(), b->blockCenters.begin(), b->blockCenters.end());
        a->allBlocksAvgCenter = getAvgPoint(a->blockCenters);
        a->updateIsSingleSegment();
        a->borderPoints.insert(a->borderPoints.end(), b->borderPoints.begin(), b->borderPoints.end());
        a->removeNext(b);
        for (auto &nextGroup : b->getNexts())
        {
            a->checkAddNext(nextGroup);
            nextGroup->checkAddNext(a);
        }
    }

    void mergeGroupsLessThanSpan(vector<shared_ptr<Group>> &groups, int span, bool useBlockCenter = false, int sizeLimit = -1)
    {
        for (int i = 0; i < groups.size(); ++i)
        {
            if (sizeLimit > 0 && groups[i]->points.size() >= sizeLimit)
            {
                continue;
            }
            for (int j = 0; j < groups.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                bool result = false;
                if (useBlockCenter)
                {
                    result = blockCentersDistance(groups[i], groups[j]) <= span;
                }
                else
                {
                    result = distanceBorderGroup(groups[i], groups[j]) <= span;
                }
                if (result)
                {
                    mergeGroup(groups[j], groups[i]);
                    groups.erase(groups.begin() + i);
                    --i;
                    break;
                }
            }
        }
    }

    /// 按颜色进行分组
    vector<shared_ptr<Group>> groupByColor(const cv::Mat &image, cv::Vec3b color, int span, int tolerance = 0)
    {
        // 定义颜色的上下限 (BGR格式)
        cv::Scalar lower_bound(std::max(0, static_cast<int>(color[0]) - tolerance),
                               std::max(0, static_cast<int>(color[1]) - tolerance),
                               std::max(0, static_cast<int>(color[2]) - tolerance));
        cv::Scalar upper_bound(std::min(255, static_cast<int>(color[0]) + tolerance),
                               std::min(255, static_cast<int>(color[1]) + tolerance),
                               std::min(255, static_cast<int>(color[2]) + tolerance));

        // 创建掩码，保留指定颜色范围内的像素
        cv::Mat original_mask;
        cv::inRange(image, lower_bound, upper_bound, original_mask);

        // 查找连通区域
        int num_labels = 0;
        cv::Mat labels, stats, centroids;
        num_labels = cv::connectedComponentsWithStats(original_mask, labels, stats, centroids);

        std::vector<std::vector<cv::Point>> contours;

        // 对每个非背景标签，查找轮廓并映射回原始掩码，跳过背景标签0
        for (int label = 1; label < num_labels; ++label)
        {
            // 创建二值掩码，仅包含当前标签
            cv::Mat currentLabelMask = (labels == label);
            // 获取currentLabelMask的所有点
            std::vector<cv::Point> points;
            cv::findNonZero(currentLabelMask, points);

            // 将找到的轮廓映射回原始掩码上的位置
            std::vector<cv::Point> originalContour;
            for (const auto &point : points)
            {
                if (original_mask.at<uchar>(point) != 0)
                {
                    originalContour.push_back(point);
                }
            }
            if (!originalContour.empty())
            {
                contours.push_back(originalContour);
            }
        }

        vector<shared_ptr<Group>> groups;
        vector<int> areas;
        for (auto &contour : contours)
        {
            auto group = make_shared<Group>(color, contour, span);
            groups.push_back(group);
            areas.push_back((int)contour.size());
        }

        if (span > 0)
        {
            // 求连通面积的中位数
            int median_area = getMedian(areas);
            int target_area = median_area * 0.7;

            vector<shared_ptr<Group>> less_groups;
            vector<shared_ptr<Group>> greater_groups;
            for (auto &group : groups)
            {
                group->updateMaxAcrossCount(median_area);
                if (group->points.size() < target_area)
                {
                    less_groups.push_back(group);
                }
                else
                {
                    greater_groups.push_back(group);
                }
            }

            // 合并less_groups中距离小于span的分组（先合并小的，不然分块太小会被忽略掉）(因为像素少，使用像素间比较)
            mergeGroupsLessThanSpan(less_groups, ceil(span * 2), false);
            greater_groups.insert(greater_groups.end(), less_groups.begin(), less_groups.end());
            // 合并总的greater_groups中面积较小的，距离小于span的分组（使用分块间比较）
//            mergeGroupsLessThanSpan(greater_groups, static_cast<int>(span * 2.2), true, median_area * 0.8);
            mergeGroupsLessThanSpan(greater_groups, static_cast<int>(span * 2), false, median_area * 0.8);
            groups = greater_groups;
        }

        vector<shared_ptr<Group>> result;
        for (auto &group : groups)
        {
            // 过滤点太少的，视为干扰（红绿相间会产生黄色）
            if (group->points.size() < span * 3)
            {
                continue;
            }
            result.push_back(group);
        }
        return result;
    }

    /**
     * @brief 使用k-means算法对给定点进行聚类。
     *
     * @param points 输入点的集合，每个点都是cv::Point类型。
     * @param clusterCount 指定想要得到的簇数量。
     * @return std::vector<cv::Point> 包含簇中心点的向量。
     */
    std::vector<cv::Point> kmeansCluster(const std::vector<cv::Point> &points, int clusterCount)
    {
        if (points.empty() || clusterCount <= 0)
        {
            LOGW(TAG, "Kmeans cluster points is empty.");
            return {};
        }
        std::vector<float> data;
        for (const auto &pt : points)
        {
            data.push_back(static_cast<float>(pt.x));
            data.push_back(static_cast<float>(pt.y));
        }
        if (data.empty())
        {
            LOGE(TAG, "Data vector is empty after conversion.");
            return {};
        }

        // 创建Mat对象并调整其形状、每个点一行，两个坐标列
        cv::Mat samples = cv::Mat(data).reshape(1, static_cast<int>(points.size()));

        if (samples.empty())
        {
            LOGE(TAG, "Failed to create Mat from data.");
            return {};
        }

        std::vector<int> labels; // 存储每个点对应的簇标签
        cv::Mat centers;         // 存储最终的簇中心

        // 执行k-means聚类
        cv::kmeans(samples, clusterCount, labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
                   3,                     // 尝试不同的初始化中心3次
                   cv::KMEANS_PP_CENTERS, // 使用k-means++算法选择初始中心
                   centers);

        // 将centers转换回std::vector<cv::Point>
        std::vector<cv::Point> centerPoints;
        for (int i = 0; i < centers.rows; ++i)
        {
            centerPoints.emplace_back(centers.at<float>(i, 0), centers.at<float>(i, 1));
        }

        return centerPoints;
    }

    /// 获取点到两点连成的直线之间的距离
    float pointToLineDistance(Point p, Point p1, Point p2)
    {
        if (p1.x == p2.x && p1.y == p2.y)
        {
            return sqrt(pow(p.x - p1.x, 2) + pow(p.y - p1.y, 2));
        }

        float a = p2.y - p1.y;
        float b = p1.x - p2.x;
        float c = p2.x * p1.y - p1.x * p2.y;
        return std::abs(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
    }

    /// 获取距离直线距离小于阈值的点
    vector<Point> getPointsCloseToLine(vector<Point> points, Point p1, Point p2, float distanceThreshold)
    {
        vector<Point> result;
        for (const auto &point : points)
        {
            if (pointToLineDistance(point, p1, p2) < distanceThreshold)
            {
                result.push_back(point);
            }
        }
        // 将result排序
        std::sort(result.begin(), result.end(), [&](const Point &a, const Point &b)
                  { return distancePoint(a, p1) < distancePoint(b, p1); });
        return result;
    }
}

std::vector<std::shared_ptr<GroupH61DX>> GroupUtilH61DX::group(cv::Mat &image, int span)
{
    // 按颜色把相同颜色的相邻像素点归为一组
    auto redGroup = groupByColor(image, Vec3b(0, 0, 255), span);
    auto greenGroup = groupByColor(image, Vec3b(0, 255, 0), span);
    auto blueGroup = groupByColor(image, Vec3b(255, 0, 0), span);
    auto yellowGroup = groupByColor(image, Vec3b(0, 255, 255), span);

    // 合并所有分组
    auto allGroups = std::vector<std::shared_ptr<Group>>();
    allGroups.insert(allGroups.end(), redGroup.begin(), redGroup.end());
    allGroups.insert(allGroups.end(), greenGroup.begin(), greenGroup.end());
    allGroups.insert(allGroups.end(), blueGroup.begin(), blueGroup.end());
    allGroups.insert(allGroups.end(), yellowGroup.begin(), yellowGroup.end());

    // 检查所有分组之间的距离，若小于等于span，如果有则添加到nexts中
    int index = 0;
    for (auto &group : allGroups)
    {
        group->tag = index;
        for (auto &next : allGroups)
        {
            if (group == next)
            {
                continue;
            }
            if (distanceBorderGroup(group, next) <= span * 2)
            {
                group->addNext(next);
            }
        }
        index++;
    }

    return allGroups;
}

cv::Point GroupUtilH61DX::findFirst(cv::Mat &image)
{
    if (image.empty())
    {
        return cv::Point(-1, -1);
    }

    // 从图像正中心向周围扩大的方式，查找第一个非0的颜色(以避免找到到图像边缘的噪点或其他干扰)
    auto center = cv::Point(image.cols / 2, image.rows / 2);
    auto end = min(image.cols, image.rows) / 2;
    for (int i = 0; i < end; ++i)
    {
        auto locations = getRangeLocations(center, i);
        for (auto &location : locations)
        {
            if (image.at<cv::Vec3b>(location)[0] != 0)
            {
                return location;
            }
        }
    }
    return cv::Point(-1, -1);
}

int GroupUtilH61DX::getSpan(cv::Mat &image, const cv::Point &start)
{
    if (image.empty())
    {
        return 3;
    }
    // 获取灯带的宽度中位数作为最大间距
    auto allWidth = std::vector<int>();
    auto rows = image.rows;
    auto cols = image.cols;
    auto first = start;
    if (first.x == -1)
    {
        // 若没有指定查询起点，则以图像中心点开始查询
        first = cv::Point(cols / 2, rows / 2);
    }

    // 统计first所在行列非零像素点的连续个数，然后取中位数
    auto count = 0;
    for (int i = 0; i < rows; i++)
    {
        auto point = cv::Point(first.x, i);
        auto rgb = getColorRGB(image, point);
        if (rgb == 0)
        {
            if (count > 0)
            {
                allWidth.push_back(count);
            }
            count = 0;
        }
        else
        {
            count++;
        }
    }
    if (count > 0)
    {
        allWidth.push_back(count);
    }

    count = 0;
    for (int i = 0; i < cols; i++)
    {
        auto point = cv::Point(i, first.y);
        auto rgb = getColorRGB(image, point);
        if (rgb == 0)
        {
            if (count > 0)
            {
                allWidth.push_back(count);
            }
            count = 0;
        }
        else
        {
            count++;
        }
    }
    if (count > 0)
    {
        allWidth.push_back(count);
    }
    return max(3, getMedian(allWidth));
}

void GroupH61DX::slicingBlock()
{
    auto count = max(2, (int)points.size() / max(1, blockWidth * blockWidth));
    if (points.size() <= count)
    {
        blockCenters = points;
        return;
    }
    blockCenters = kmeansCluster(points, count);
    allBlocksAvgCenter = getAvgPoint(blockCenters);
}

void GroupH61DX::calculateBorder()
{
    struct PointHash
    {
        std::size_t operator()(const cv::Point &p) const
        {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };
    auto border = std::vector<cv::Point>();
    std::unordered_set<cv::Point, PointHash> borderSet(points.begin(), points.end());
    for (auto &point : points)
    {
        // 查看上下左右是否都有元素，若不是则加入
        auto up = cv::Point(point.x, point.y - 1);
        auto down = cv::Point(point.x, point.y + 1);
        auto left = cv::Point(point.x - 1, point.y);
        auto right = cv::Point(point.x + 1, point.y);
        if (borderSet.find(up) == borderSet.end() || borderSet.find(down) == borderSet.end() || borderSet.find(left) == borderSet.end() || borderSet.find(right) == borderSet.end())
        {
            border.push_back(point);
        }
    }
    borderPoints = vector<cv::Point>(border.begin(), border.end());
}

void GroupH61DX::updateMaxAcrossCount(int avgPointsCount)
{
    maxAcrossCount = static_cast<int>(points.size()) / avgPointsCount + 1;
    areaRate = static_cast<double>(points.size())/static_cast<double>(avgPointsCount);
}

void GroupH61DX::updateIsSingleSegment()
{
    if (blockCenters.size() < 3) {
        isSingleSegment = true;
        return;
    }
    auto start = blockCenters[0];
    auto end = blockCenters[1];
    // 如果任意两点都能经过所有分块，则是单段
    auto ps = getPointsCloseToLine(blockCenters, start, end, static_cast<float>(max(blockWidth, blockWidth -1)));
    isSingleSegment = blockCenters.size() == ps.size();
}

int GroupH61DX::distanceWithGroup(std::shared_ptr<GroupH61DX> group, bool useBlockCenter)
{
    if (useBlockCenter)
    {
        return blockCentersDistance(getSharedPtr(), group);
    }
    return distanceBorderGroup(getSharedPtr(), group);
}

/// 合并两个分组
void GroupH61DX::merge(std::shared_ptr<GroupH61DX> other)
{
    mergeGroup(getSharedPtr(), other);
}

cv::Point GroupH61DX::getClosestBlockCenter(const std::shared_ptr<GroupH61DX> &group)
{
    auto minDistance = INT_MAX;
    auto minIndex = -1;
    for (int i = 0; i < blockCenters.size(); i++)
    {
        for (auto &point : group->blockCenters)
        {
            auto distance = distancePoint(blockCenters[i], point);
            if (distance < minDistance)
            {
                minDistance = distance;
                minIndex = i;
            }
        }
    }
    if (minIndex == -1)
    {
        return cv::Point(-1, -1);
    }
    return blockCenters[minIndex];
}

std::vector<cv::Point> GroupH61DX::getPathCenters(const std::shared_ptr<GroupH61DX> &fromGroup, const std::shared_ptr<GroupH61DX> &toGroup, int span)
{
    if (this->blockCenters.size() == 0)
    {
        return {};
    }

    if (fromGroup && toGroup)
    {
        auto fromPoint = getClosestBlockCenter(fromGroup);
        auto toPoint = getClosestBlockCenter(toGroup);
        return getPointsCloseToLine(this->blockCenters, fromPoint, toPoint, span);
    }
    else if (fromGroup)
    {
        auto fromPoint = getClosestBlockCenter(fromGroup);
        auto toPoint = fromGroup->getClosestBlockCenter(this->getSharedPtr());
        return getPointsCloseToLine(this->blockCenters, fromPoint, toPoint, span);
    }
    else if (toGroup)
    {
        auto fromPoint = toGroup->getClosestBlockCenter(this->getSharedPtr());
        auto toPoint = getClosestBlockCenter(toGroup);
        return getPointsCloseToLine(this->blockCenters, fromPoint, toPoint, span);
    }
    return {this->blockCenters[0]};
}

std::vector<std::shared_ptr<GroupH61DX>> GroupH61DX::pickNexts(int rgb, std::shared_ptr<GroupH61DX> fromGroup)
{
    // 过滤出rgb相同的
    auto sameRgb = std::vector<std::shared_ptr<GroupH61DX>>();
    for (auto &next : nexts)
    {
        auto nextLock = next.lock();
        if (nextLock != fromGroup && nextLock->rgb == rgb)
        {
            sameRgb.push_back(nextLock);
        }
    }
    if (sameRgb.size() <= 1 || fromGroup == nullptr)
    {
        return sameRgb;
    }
    
    

    // 找与直线同方向的
    struct TempResult {
        std::shared_ptr<GroupH61DX> group;
        double angle;
    };
    auto temps = std::vector<TempResult>();
    auto angles = std::vector<double>();
    auto startPoint = fromGroup->allBlocksAvgCenter;
    auto endPoint = this->allBlocksAvgCenter;
    
    // 如果当前段是单段或下一段是单段，则使用该段进行连线判断
//    shared_ptr<GroupH61DX> single = nullptr;
//    if (isSingleSegment && blockCenters.size() > 1) {
//        single = getSharedPtr();
//    } else if (fromGroup->isSingleSegment && fromGroup->blockCenters.size() > 1) {
//        single = fromGroup;
//    }
//    
//    if (single) {
//        std::sort(single->blockCenters.begin(), single->blockCenters.end(), [&](const Point &t1, const Point &t2)
//                              { return distancePoint(t1, startPoint) < distancePoint(t2, startPoint); });
//        startPoint = single->blockCenters[0];
//        endPoint = single->blockCenters[1];
//    }
    
    for (auto &next : sameRgb)
    {
        auto angle = getTargetPointAngle(startPoint, endPoint, next->allBlocksAvgCenter);
        if (angle < CV_PI / 4)
        {
            temps.push_back({ next, angle });
        }
    }
    std::sort(temps.begin(), temps.end(), [](const TempResult& a, const TempResult& b) {
        return a.angle < b.angle;
    });
    
    auto result = std::vector<std::shared_ptr<GroupH61DX>>();
    for (TempResult& t : temps) {
        result.push_back(t.group);
    }
    
//    if (result.size() > 1 && this->blockCenters.size() > 1)
//    {
//        // 如果有多个结果，当前最末尾分块的两个分块心点，和目标任意当前的两个分块中心进行角度比较，取小的那一个
//        auto a = this->blockCenters[0];
//        auto b = this->blockCenters[1];
//        auto nowPoints = getPointsCloseToLine(this->blockCenters, startPoint, endPoint, 10);
//        if (nowPoints.size() > 1)
//        {
//            // 如果是大面积的分组（即多段重叠的），取前面两个
//            if (areaRate > 1.2) {
//                a = nowPoints[0];
//                b = nowPoints[1];
//            } else {
//                a = nowPoints[nowPoints.size() - 2];
//                b = nowPoints[nowPoints.size() - 1];
//            }
//        }
//        double minAngle = DBL_MAX;
//        shared_ptr<GroupH61DX> target = nullptr;
//        for (auto &next : result)
//        {
//            auto blocks = next->blockCenters;
//            // 对blocks排序，取最近b点的一个
//            std::sort(blocks.begin(), blocks.end(), [&](const Point &t1, const Point &t2)
//                      { return distancePoint(t1, b) < distancePoint(t2, b); });
//            auto angle = getTargetPointAngle(a, b, blocks[0]);
//            if (angle < minAngle)
//            {
//                minAngle = angle;
//                target = next;
//            }
//        }
//        if (target != nullptr)
//        {
//            return {target};
//        }
//    }
    return result;
}

#if DEBUG
void GroupH61DX::debugPrint()
{
    // 打印自己的所有点和nexts的所有点
    LOGD(TAG, "color: %c, points: %d, nexts: %d, tag: %d", this->debugChar(), points.size(), nexts.size(), tag);
    // 使用A打印自己所有点，next使用其debugChar打印其所有点，用占位符代替
    const char placeHolder = ' ';
    auto allLines = std::vector<std::vector<char>>();
    auto width = 120;
    // 初始化alllines为全*
    for (int i = 0; i < width; i++)
    {
        auto line = std::vector<char>(width, placeHolder);
        allLines.push_back(line);
    }
    // 以points[0]为中心点，开始打印
    auto left = points[0].x - 50;
    auto top = points[0].y - 50;
    for (auto &point : points)
    {
        auto x = point.x - left;
        auto y = point.y - top;
        if (x >= 0 && x < width && y >= 0 && y < width)
        {
            allLines[y][x] = 'A';
        }
    }
    auto nexts = getNexts();
    for (auto &next : nexts)
    {
        auto ps = next->points;
        for (auto &point : ps)
        {
            auto x = point.x - left;
            auto y = point.y - top;
            if (x >= 0 && x < width && y >= 0 && y < width)
            {
                allLines[y][x] = next->debugChar();
            }
        }
    }
    // 将allLines添加换行符并打印出来
    auto line = std::string();
    for (auto &l : allLines)
    {
        for (auto &c : l)
        {
            line += c;
        }
        line += "\n";
    }
    cout << line;
}

#endif
