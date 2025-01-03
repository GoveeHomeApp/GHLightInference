#include "DetectionH61DX.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "ColorCodingH61DX.hpp"
#include "PictureH61DX.hpp"
#include "GroupUtilH61DX.hpp"
#include <unordered_set>
#include <unordered_map>
#include "LogUtils.h"

using namespace cv;
using namespace std;
using namespace H61DX;

namespace {
    /// 获取所有分组的标识
    vector<int> getGroupTags(vector<shared_ptr<GroupH61DX>>& groups) {
        vector<int> tags;
        for (auto &group : groups) {
            tags.push_back(group->tag);
        }
        return tags;
    }

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
                        if (count >= 2) {
                            startGroups.push_back(group);
                            break;
                        }
                    }
                }
            }
        }
        return startGroups;
    }

#if DEBUG
    /// 将所有char连接起来一起打印，3个一个空格
    void printGroupChars(std::vector<std::shared_ptr<GroupH61DX>>& groups) {
        auto rgbSort = std::string();
        for (size_t i = 0; i < groups.size(); ++i) {
            auto &group = groups[i];
            rgbSort += group->debugChar();
            if (i % 3 == 2) {
                rgbSort += " ";
            }
        }
        cout << rgbSort << endl;
    }
#endif

    int checkAcrossCount(std::unordered_map<int, int>& across, const shared_ptr<GroupH61DX>& group)
    {
        auto value = across.find(group->tag);
        if (value != across.end()) {
            return value->second;
        }
        return 0;
    }

    bool isCanAcrossGroup(std::unordered_map<int, int>& across, const shared_ptr<GroupH61DX>& group)
    {
        auto max = group->maxAcrossCount;
        auto count = checkAcrossCount(across, group);
        return count < max;
    }
    
    void acrossGroup(std::unordered_map<int, int>& across, const shared_ptr<GroupH61DX>& group) {
        across[group->tag] = checkAcrossCount(across, group) + 1;
    }

    /// 统计不同实例数量的函数
    template<typename T>
    size_t count_unique_instances(const std::vector<std::shared_ptr<T>>& vec) {
        std::unordered_set<const void*> uniqueInstances;
        for (const auto& ptr : vec) {
            if (ptr) {
                uniqueInstances.insert(ptr.get());
            }
        }
        return uniqueInstances.size();
    }

    /// 比较两个分组，a是否比b好
    bool compare(const vector<shared_ptr<GroupH61DX>>& a, const vector<shared_ptr<GroupH61DX>>& b) {
        if (a.size() > b.size()) {
            return true;
        } else if (a.size() == b.size()) {
            return count_unique_instances(a) > count_unique_instances(b);
        }
        return false;
    }

    vector<shared_ptr<GroupH61DX>> sort(const shared_ptr<GroupH61DX>& group, 
                                        const vector<int>& colors,
                                        int index = 0,
                                        vector<shared_ptr<GroupH61DX>> nowGroups = {},
                                        const shared_ptr<GroupH61DX>& from = nullptr,
                                        std::unordered_map<int, int> across = {}) {
        if (group == nullptr || index >= colors.size() || group->rgb != colors[index]) {
            return nowGroups;
        }
        nowGroups.push_back(group);
        acrossGroup(across, group);

        #if DEBUG
            printGroupChars(nowGroups);
            auto tags = getGroupTags(nowGroups);
            for (auto &tag : tags) {
                printf("%d ", tag);
            }
            printf("\n");
        #endif

        auto result = nowGroups;
        if (index < colors.size() - 1)
        {
            auto nextColor = colors[index + 1];
            auto nexts = group->pickNexts(nextColor, from);
            for (auto &next : nexts) {
                if (from != nullptr && from == next && isCanAcrossGroup(across, group)) 
                {
                    continue;
                }
                auto nextGroups = sort(next, colors, index + 1, nowGroups, group, across);
                if (compare(nextGroups, result))
                {
                    result = nextGroups;
                }
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
    _nowImage = PictureH61DX::debugProcessImage(_originImage, [&callback](auto image){
        callback({image});
    });
    
    auto first = GroupUtilH61DX::findFirst(_nowImage);
    if (first.x == -1)
    {
        LOGE(TAG, "Find first failed");
        return {};
    }

    auto span = GroupUtilH61DX::getSpan(_nowImage, first);
    LOGD(TAG, "---> Span: %d", span);

    auto all = GroupUtilH61DX::group(_nowImage, span);
    auto startGroups = getStartGroup(all);
    if (startGroups.size() == 0) {
        LOGE(TAG, "Start group is empty");
        return {};
    }
    
    LOGD(TAG, "<<--------------------- all group --------------------->>");
    for (auto &group : all) {
        group->debugPrint();
    }
    
    // 找到group的rgb值与_detectionColors一致的序列
    auto resutlGroups = sort(startGroups[0], _detectionColors);
    for (size_t i = 1; i < startGroups.size(); ++i)
    {
        auto groups = sort(startGroups[i], _detectionColors);
        if (compare(groups, resutlGroups))
        {
            resutlGroups = groups;
        }
    }
    if (resutlGroups.size() == 0) {
        LOGE(TAG, "Result group is empty");
        return {};
    }
    
    // 打印排序结果
    LOGD(TAG, "<<--------------------- final result --------------------->>");
    for (auto &group : resutlGroups) {
        group->debugPrint();
    }
    printGroupChars(resutlGroups);

    auto centers = vector<cv::Point>();
    shared_ptr<GroupH61DX> last = nullptr;
    auto end = resutlGroups.size();
    for (int i = 0; i < end; ++i) {
        auto now = resutlGroups[i];
        shared_ptr<GroupH61DX> next = nullptr;
        if (i + 1 < end) {
            next = resutlGroups[i + 1];
        }
        auto points = now->getPathCenters(last, next, span);
        centers.insert(centers.end(), points.begin(), points.end());
        last = now;
    }
    
    // 新建一个图片，将centers连接起来
    for (size_t i = 1; i < centers.size(); ++i) {
        auto &point = centers[i - 1];
        auto &nextPoint = centers[i];
        cv::line(_nowImage, point, nextPoint, cv::Scalar(255, 0, 255), 2);
    }
    callback({ _nowImage });

    return centers;
}

#endif
