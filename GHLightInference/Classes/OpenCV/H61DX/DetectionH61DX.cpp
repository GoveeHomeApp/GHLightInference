#include "DetectionH61DX.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "ColorCodingH61DX.hpp"
#include "PictureH61DX.hpp"
#include "GroupUtilH61DX.hpp"

using namespace cv;
using namespace std;
using namespace H61DX;

namespace {
    /// 获取开始的可能分组
    vector<shared_ptr<GroupH61DX>> getStartGroup(vector<shared_ptr<GroupH61DX>> groups) {
        vector<shared_ptr<GroupH61DX>> startGroups;
        // 如果是黄色分组只有一个绿色next，直接返回它，否则加入到startGroups
        for (auto &group : groups) {
            if (group->rgb != COLOR_YELLOW) {
                continue;
            }
            
#if DEBUG
            group->debugPrint();
#endif
            
            auto nexts = group->getNexts();
            if (nexts.size() == 1 && nexts[0]->rgb == COLOR_GREEN) {
                return { group };
            } else if (nexts.size() > 1) {
                // 如果nexts中包含绿色，且绿色分组中包含两个黄色，则将其加入
                for (auto &next : nexts) {
                    if (next->rgb == COLOR_GREEN) {
#if DEBUG
                        next->debugPrint();
#endif
                        auto greenNexts = next->getNexts();
                        auto count = 0;
                        for (auto &greenNext : greenNexts) {
                            if (greenNext->rgb == COLOR_YELLOW) {
                                count++;
                            }
                        }
                        if (count == 2) {
                            startGroups.push_back(group);
                            break;
                        }
                    }
                }
            }
        }
        return startGroups;
    }

    vector<shared_ptr<GroupH61DX>> sort(const shared_ptr<GroupH61DX>& group, const vector<int>& colors, int index = 0, vector<shared_ptr<GroupH61DX>> nowGroups = {}, const shared_ptr<GroupH61DX>& from = nullptr) {
        if (group == nullptr || index >= colors.size() || group->rgb != colors[index]) {
            // 如果走完了的，则统计起来，取所有格子都走过的那个
            // if (index == colors.size() - 1)
            // {
                
            // }
            
            return nowGroups;
        }
        nowGroups.push_back(group);
        auto result = nowGroups;
        for (auto &next : group->getNexts()) {
            if (from != nullptr && from == next) {
                continue;
            }
            auto nextGroups = sort(next, colors, index + 1, nowGroups);
            if (nextGroups.size() > result.size()) {
                result = nextGroups;
            }
        }
        
        return result;
    }

    cv::Point getGroupCenter(const shared_ptr<GroupH61DX>& group) {
        auto allX = 0, allY = 0;
        auto points = group->points;
        auto count = points.size();
        for (auto &point : points) {
            allX += point.x;
            allY += point.y;
        }
        return cv::Point(allX / count, allY / count);
    }
}

DetectionH61DX::DetectionH61DX(int icCount) : icCount(icCount)
{
    _detectionColors = ColorCodingH61DX(icCount).getDetectionColors();
}

DetectionH61DX::~DetectionH61DX()
{
    _originImage.release();
    _nowImage.release();
}

void DetectionH61DX::detection(cv::Mat originImage, std::function<void(std::vector<cv::Point>)> callback)
{
    _originImage = originImage.clone();
    _callback = callback;
}

#if DEBUG

std::vector<cv::Point> DetectionH61DX::debugDetection(cv::Mat originImage, std::function<void(std::vector<cv::Mat>)> callback)
{
    cvtColor(originImage, _originImage, COLOR_RGBA2BGR);
    _nowImage = PictureH61DX::processImage(_originImage);
    callback({ _originImage, _nowImage });
    
    auto group = GroupUtilH61DX::group(_nowImage);
    auto startGroups = getStartGroup(group);
    if (startGroups.size() == 0) {
        LOGE(TAG, "Start group is empty");
        return {};
    }
    
    // 找到group的rgb值与_detectionColors一致的序列
    auto resutlGroups = sort(startGroups[0], _detectionColors);
    for (size_t i = 1; i < startGroups.size(); ++i)
    {
        auto groups = sort(startGroups[i], _detectionColors);
        if (groups.size() > resutlGroups.size()) {
            resutlGroups = groups;
        }
    }
    if (resutlGroups.size() == 0) {
        LOGE(TAG, "Result group is empty");
        return {};
    }
    
    // 打印排序结果
    for (auto &group : resutlGroups) {
        group->debugPrint();
    }
    // 将所有char连接起来一起打印，3个一个空格
    auto rgbSort = std::string();
    for (size_t i = 0; i < resutlGroups.size(); ++i) {
        auto &group = resutlGroups[i];
        rgbSort += group->debugChar();
        if (i % 3 == 2) {
            rgbSort += " ";
        }
    }
    cout << rgbSort << endl;

    auto centers = vector<cv::Point>();
    for (auto &group : resutlGroups) {
        centers.push_back(getGroupCenter(group));
    }

    // 新建一个图片，将centers连接起来
    for (size_t i = 1; i < centers.size(); ++i) {
        auto &point = centers[i - 1];
        auto &nextPoint = centers[i];
        cv::line(_nowImage, point, nextPoint, cv::Scalar(255, 0, 255), 3);
    }
    callback({ _nowImage });

    return centers;
}

#endif
