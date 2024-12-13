#include "GroupUtilH61DX.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "UtilH61DX.hpp"

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

    /// 查找color对应的所有联通区域，并返回原图中的坐标点
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

        // 如果span大于0，则在副本上进行膨胀操作以连接间隔不超过span的像素
        cv::Mat dilated_mask = original_mask.clone();
        if (span > 0)
        {
            // 创建一个尺寸为 (2*span + 1) x (2*span + 1) 的结构元素
            cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * span + 1, 2 * span + 1));
            cv::dilate(dilated_mask, dilated_mask, structuringElement);
        }

        // 查找连通区域
        int num_labels = 0;
        cv::Mat labels, stats, centroids;
        num_labels = cv::connectedComponentsWithStats(dilated_mask, labels, stats, centroids);

        // 存储结果的容器
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
        for (auto &contour : contours)
        {
            // 过滤点太少的，视为干扰（红绿相间会产生黄色）
            if (contour.size() < span * 2) {
                continue;
            }
            auto group = make_shared<Group>(color, contour);
            groups.push_back(group);
        }
        return groups;
    }

    /// 计算两点之间的距离，八个方向
    int distancePoint(Point a, Point b) {
        return max(abs(a.x - b.x), abs(a.y - b.y));
    }

    /// 计算两个组合之间的距离，取最小距离
    int distanceGroup(std::shared_ptr<Group> a, std::shared_ptr<Group> b) {
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
}

std::vector<std::shared_ptr<GroupH61DX>> GroupUtilH61DX::group(cv::Mat &image)
{
    auto first = findFirst(image);
    if (first.x == -1)
    {
        LOGE(TAG, "Find first failed");
        return {};
    }
    
    auto span = GroupUtilH61DX::getSpan(image, first);

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
    for (auto &group : allGroups)
    {
        for (auto &next : allGroups)
        {
            if (group == next)
            {
                continue;
            }
            if (distanceGroup(group, next) <= span)
            {
                group->addNext(next);
            }
        }
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
    return max(3, getMedian(allWidth)/3*2);
}

#if DEBUG
void GroupH61DX::debugPrint() {
    // 打印自己的所有点和nexts的所有点
    LOGD(TAG, "color: %c, points: %d, nexts: %d", this->debugChar(), points.size(), nexts.size());
    // 使用A打印自己所有点，next使用其debugChar打印其所有点，用占位符代替
    const char placeHolder = ' ';
    auto allLines = std::vector<std::vector<char>>();
    auto width = 120;
    // 初始化alllines为全*
    for (int i = 0; i < width; i++) {
        auto line = std::vector<char>(width, placeHolder);
        allLines.push_back(line);
    }
    // 以points[0]为中心点，开始打印
    auto left = points[0].x - 50;
    auto top = points[0].y - 50;
    for (auto &point : points) {
        auto x = point.x - left;
        auto y = point.y - top;
        if (x >= 0 && x < width && y >= 0 && y < width) {
            allLines[y][x] = 'A';
        }
    }
    auto nexts = getNexts();
    for (auto &next : nexts) {
        auto ps = next->points;
        for (auto &point : ps) {
            auto x = point.x - left;
            auto y = point.y - top;
            if (x >= 0 && x < width && y >= 0 && y < width) {
                allLines[y][x] = next->debugChar();
            }
        }
    }
    // 将allLines添加换行符并打印出来
    auto line = std::string();
    for (auto &l : allLines) {
        for (auto &c : l) {
            line += c;
        }
        line += "\n";
    }
    cout << line;
}

#endif
