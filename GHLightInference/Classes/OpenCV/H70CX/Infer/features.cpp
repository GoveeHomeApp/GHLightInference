#include "features.hpp"
#include "aligner2.cpp"
#include "LogUtils.h"
#include "EnhancedLookup.cpp"
#include <sstream>
#include <iomanip>
#include <csetjmp>

// 局部变量
namespace {
    int lightType = 0;
    vector<Point2f> pPointXys;
    unordered_map<int, Mat> frameStepMap;
    vector<LightPoint> errorSerialVector;
    unordered_map<int, vector<LightPoint>> sequenceTypeMap;
    EnhancedChristmasTreeAligner aligner;
}

Mat alignResize(int frameStep, Mat &originalMat, vector<Mat> &outMats) {
    if (originalMat.empty()) {
        LOGE(LOG_TAG, "Empty input matrix");
        return {};
    }

    Mat srcResize;
    const Size newSize(640, 640);

    try {
        if (frameStep > STEP_VALID_FRAME_START) {
            auto it = frameStepMap.find(STEP_VALID_FRAME_START);
            if (it == frameStepMap.end()) {
                LOGE(LOG_TAG, "Reference frame not found");
                return {};
            }

            Mat alignMat = aligner.alignImage(it->second, originalMat, outMats);
            if (alignMat.empty()) {
                return alignMat;
            }

            resize(alignMat, srcResize, newSize);
            frameStepMap[frameStep] = alignMat;
        } else {
            release();
            aligner = EnhancedChristmasTreeAligner();
            resize(originalMat, srcResize, newSize);
            frameStepMap[frameStep] = originalMat;
        }

        return srcResize;
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "Error in alignResize: %s", e.what());
        return {};
    }
}

void mergeNearbyPoints(vector<Point2f> &points, float threshold) {
    if (points.empty()) return;

    vector<Point2f> merged_points;
    vector<bool> used(points.size(), false);

    for (size_t i = 0; i < points.size(); i++) {
        if (used[i]) continue;

        // 用于累积需要合并的点的坐标
        float sum_x = points[i].x;
        float sum_y = points[i].y;
        int count = 1;
        used[i] = true;

        // 查找所有与当前点距离小于阈值的点
        for (size_t j = i + 1; j < points.size(); j++) {
            if (used[j]) continue;

            float dist = norm(points[i] - points[j]);
            if (dist < threshold) {
                sum_x += points[j].x;
                sum_y += points[j].y;
                count++;
                used[j] = true;
                LOGD(LOG_TAG, "合并点位: (%.2f, %.2f) -> (%.2f, %.2f), 距离: %.2f",
                     points[j].x, points[j].y, points[i].x, points[i].y, dist);
            }
        }

        // 如果找到需要合并的点，计算平均位置
        if (count > 1) {
            Point2f merged_point(sum_x / count, sum_y / count);
            merged_points.push_back(merged_point);
            LOGW(LOG_TAG, "合并%d个点位为: (%.2f, %.2f)",
                 count, merged_point.x, merged_point.y);
        } else {
            merged_points.push_back(points[i]);
        }
    }

    // 更新原始点集
    points = merged_points;
    LOGD(LOG_TAG, "合并后点位数量: %zu", points.size());
}

float calculateAverageCircleDiameter(const vector<LightPoint> &points) {
    if (points.empty()) {
        return 8.0f; // 默认值
    }

    float totalDiameter = 0.0f;
    int validCount = 0;

    for (const auto &point: points) {
        // 使用矩形的短边作为直径
        float diameter = min(point.tfRect.width, point.tfRect.height);
        if (diameter > 0) {
            totalDiameter += diameter;
            validCount++;
        }
    }

    if (validCount == 0) {
        return 8.0f; // 默认值
    }

    float avgDiameter = totalDiameter / validCount;
    LOGD(LOG_TAG, "Average circle diameter: %.2f", avgDiameter);
    return avgDiameter;
}

String sortStripByStep(int frameStep, vector<LightPoint> &resultObjects,
                       int lightTypeP, vector<Mat> &outMats) {
    lightType = lightTypeP;
    Mat image = frameStepMap[STEP_VALID_FRAME_START];
    try {
        if (frameStep == STEP_VALID_FRAME_START) {
            // 处理每个检测到的点
            for (const auto &point: resultObjects) {
                pPointXys.push_back(point.position);
            }
            LOGD(LOG_TAG, "resultObjects =%d, pPointXys=%d", resultObjects.size(),
                 pPointXys.size());
            //绘制原始定位点位 LEDDetector
            drawPointsWithCircles(image, pPointXys, outMats,
                                  "Original Points");

            // 计算合并阈值
            float avgDiameter = calculateAverageCircleDiameter(resultObjects);
            float mergeThreshold = avgDiameter * 0.7f;
            LOGD(LOG_TAG, "Merge threshold: %.2f (75%% of avg diameter: %.2f)",
                 mergeThreshold, avgDiameter);

            // 合并相邻的点
            mergeNearbyPoints(pPointXys, mergeThreshold);

            // 绘制合并后的点位
            drawPointsWithCircles(image, pPointXys, outMats, "Merged Points");

            // 移除离群点
            vector<int> eraseVector = polyPoints(pPointXys, 3, 2.3);
            sort(eraseVector.begin(), eraseVector.end(), std::greater<int>());
            for (int index: eraseVector) {
                if (pPointXys.begin() + index < pPointXys.end() - 1) {
                    pPointXys.erase(pPointXys.begin() + index);
                }
            }

            //绘制丢弃离群点后点的点位
            drawPointsWithCircles(frameStepMap[STEP_VALID_FRAME_START], pPointXys, outMats,
                                  "Points After Outlier Removal");

            try {
                // 创建检测器实例
                LEDDetector detector(image, pPointXys);
                // 分析已知灯珠特征
                detector.analyzeKnownLEDs();
                // 查找遗漏的灯珠
                vector<Point2f> missing_leds = detector.findMissingLEDs();
                LOGW(LOG_TAG, " 查找遗漏的灯珠 missing_leds = %d KnownPositions = %d",
                     missing_leds.size(), detector.getKnownPositions().size());

                if (Logger::debugSwitch) {
                    Mat result = detector.visualize();
                    outMats.push_back(result);
                }

                pPointXys.clear();
                for (const auto &item: detector.getKnownPositions()) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPointXys.push_back(item);
                }
                for (const auto &item: missing_leds) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPointXys.push_back(item);

                }
            } catch (...) {
                LOGE(LOG_TAG, "error LEDDetector");
                pPointXys.clear();
                for (const auto &item: pPointXys) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPointXys.push_back(item);
                }
            }

            processInitialFrame(pPointXys, outMats);
            pPointXys.clear();
            return "";
        }

        auto it = frameStepMap.find(frameStep);
        if (it == frameStepMap.end()) {
            LOGE(LOG_TAG, "Frame not found for step %d", frameStep);
            return "{}";
        }

        try {
            bool processResult = g_colorSplitter.processImage(it->second, outMats);
            LOGD(LOG_TAG, "processImage result: %s", processResult ? "true" : "false");
        } catch (...) {
            LOGE(LOG_TAG, "processImage error");
        }

        if (frameStep != g_colorSplitter.getMaxSplitCount() - 1 &&
            !g_colorSplitter.isAllPointsConfirmed()) {
            return "";
        }

        auto confirmedPoints = g_colorSplitter.getConfirmedPoints();
        LOGW(LOG_TAG, "原始confirmedPoints=%d", confirmedPoints.size());

        // 验证点位
        confirmedPoints = validatePointPositions(confirmedPoints, 2);

        vector<Point2f> trapezoid4Points;
        if (lightType == TYPE_H70CX_3D) {
            if (confirmedPoints.size() < 5) {
                return "{}";
            }
            Mat trapezoidMat = frameStepMap[STEP_VALID_FRAME_START].clone();
            vector<Point2f> point4Trapezoid;
            for (auto &confirmedPoint: confirmedPoints) {
                point4Trapezoid.push_back(confirmedPoint.position);
            }
            LOGD(LOG_TAG, "point4Trapezoid = %d", point4Trapezoid.size());
            int ret = getMinTrapezoid(trapezoidMat, point4Trapezoid, trapezoid4Points);
            if (ret != 1) {
                LOGE(LOG_TAG, "构建梯形异常");
            }
            if (Logger::debugSwitch) {
                outMats.push_back(trapezoidMat);
            } else {
                trapezoidMat.release();
            }
        } else {
            try {
                confirmedPoints = interpolateAndExtrapolatePoints(confirmedPoints,
                                                                  g_colorSplitter.getIcNum());
                LOGW(LOG_TAG, "confirmedPoints=%d", confirmedPoints.size());
            } catch (...) {
                LOGE(LOG_TAG, "interpolateAndExtrapolatePoints error");
            }
        }

        drawPointsWithLabels(frameStepMap[STEP_VALID_FRAME_START], confirmedPoints, outMats,
                             "Confirmed Points");

        release();
        return splicedJson(lightPointsToJson(confirmedPoints, lightTypeP),
                           point2iToJson(trapezoid4Points));
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "Error in sortStripByStep: %s", e.what());
        return "{}";
    }
}

bool compareIndex(const LedPoint &p1, const LedPoint &p2) {
    return p1.id < p2.id;
}


void processInitialFrame(vector<Point2f> pPointXys, vector<Mat> &outMats) {
    try {
        // 获取初始帧图像
        auto it = frameStepMap.find(STEP_VALID_FRAME_START);
        if (it == frameStepMap.end()) {
            LOGE(LOG_TAG, "Initial frame not found");
            return;
        }
        Mat &initialFrame = it->second;

        // 初始化颜色分割器
        if (!g_colorSplitter.initialize(initialFrame, pPointXys, outMats)) {
            LOGE(LOG_TAG, "Failed to initialize color splitter");
            return;
        }

        LOGD(LOG_TAG, "Processed initial frame with %zu points", pPointXys.size());
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "Error in processInitialFrame: %s", e.what());
    } catch (...) {
        LOGE(LOG_TAG, "Unknown error in processInitialFrame");
    }
}

void release() {
    try {
        for (auto &pair: frameStepMap) {
            pair.second.release();
        }
        frameStepMap.clear();
        sequenceTypeMap.clear();
        errorSerialVector.clear();
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "Error in release: %s", e.what());
    }
}

void drawPointsWithCircles(const Mat &src, const vector<Point2f> points, vector<Mat> &outMats,
                           const string &title) {
    try {
        if (!Logger::debugSwitch)return;
        Mat dstCircle = src.clone();

        // 绘制标题
        if (!title.empty()) {
            putText(dstCircle, title, Point(50, 50), FONT_HERSHEY_SIMPLEX,
                    0.7, Scalar(255, 0, 50), 2);
        }

        for (auto &center: points) {
            try {
                if (center.x < 8 || center.x >= src.cols - 8 || center.y < 8 ||
                    center.y >= src.rows - 8) {
                    continue;
                }
                circle(dstCircle, center, 6, Scalar(0, 0, 0, 150), 2);
            } catch (...) {}
        }
        outMats.push_back(dstCircle);
    } catch (std::exception &e) {
        LOGE(LOG_TAG, "drawPointsWithCircles e = %s", e.what());
    } catch (...) {
        LOGE(LOG_TAG, "drawPointsWithCircles error");
    }
}

void drawPointsWithLabels(const Mat &src, const vector<LedPoint> lightPoints, vector<Mat> &outMats,
                          const string &title) {
    try {
        if (!Logger::debugSwitch)return;
        Mat dstCircle = src.clone();

        // 绘制标题
        if (!title.empty()) {
            putText(dstCircle, title, Point(50, 50), FONT_HERSHEY_SIMPLEX,
                    0.7, Scalar(255, 0, 50), 2);
        }

        for (const auto &lightPoint: lightPoints) {
            try {
                Point2f center = lightPoint.position;
                if (center.x < 8 || center.x >= src.cols - 8 || center.y < 8 ||
                    center.y >= src.rows - 8) {
                    continue;
                }
                center.x = static_cast<int>(center.x);
                center.y = static_cast<int>(center.y);
                circle(dstCircle, center, 6, Scalar(255, 0, 0, 150), 1);
                putText(dstCircle, to_string(lightPoint.id), center,
                        FONT_HERSHEY_SIMPLEX,
                        0.3,
                        Scalar(0, 255, 255),
                        1);
            } catch (...) {}
        }
        outMats.push_back(dstCircle);
    } catch (std::exception &e) {
        LOGE(LOG_TAG, "drawPointsWithLabels e = %s", e.what());
    } catch (...) {
        LOGE(LOG_TAG, "drawPointsWithLabels error");
    }
}

void logLabels(const std::vector<LightPoint> &input) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < input.size(); ++i) {
        oss << input[i].label;
        if (i < input.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    LOGD(LOG_TAG, "%s", oss.str().c_str());
}


std::string floatToDouble(float value, int precision = 1) {
    auto d = static_cast<double>(value); // �� float 转换为 double
    std::ostringstream out; // 创建一个字符串流
    out << std::fixed << std::setprecision(precision) << d; // 设置格式并写入字符串流
    return out.str(); // 返回格式化后的字符串
}

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LedPoint> &points, int lightTypeSet) {
    try {
        stringstream ss;
        ss << "[";
        for (int i = 0; i < points.size(); i++) {
            ss << "{";
            ss << "\"x\": " << floatToDouble(points[i].position.x) << ", ";
            ss << "\"y\": " << floatToDouble(points[i].position.y) << ", ";
            ss << "\"index\": " << points[i].id;
            ss << "}";
            if (i < points.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    } catch (...) {
        LOGE(LOG_TAG, "lightPointsToJson error");
        return "[]";
    }
}

string splicedJson(string a, string b) {
    try {
        stringstream ss;
        ss << "{";
        if (lightType == TYPE_H70CX_3D) {
            ss << "\"lightPoints\": " << a << ", ";
            ss << "\"trapezoidalPoints\": " << b << "";
        } else {
            ss << "\"lightPoints\": " << a << "";
        }
        ss << "}";
        return ss.str();
    } catch (...) {
        LOGE(LOG_TAG, "splicedJson error");
        return "{}";
    }
}

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2f> &points) {
    try {
        stringstream ss;
        ss << "[";
        for (int i = 0; i < points.size(); i++) {
            string name;
            if (i == 0) {
                name = "rT";
            } else if (i == 1) {
                name = "rB";
            } else if (i == 2) {
                name = "lB";
            } else {
                name = "lT";
            }
            ss << "{";
            ss << "\"pName\": \"" << name << "\", ";
            ss << "\"x\": " << floatToDouble(points[i].x) << ", ";
            ss << "\"y\": " << floatToDouble(points[i].y) << "";
            ss << "}";
            if (i < points.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    } catch (...) {
        LOGE(LOG_TAG, "point2iToJson error");
        return "[]";
    }
}


Rect2i safeRect2i(const Rect2i &region, const Size &imageSize) {
    Rect2i safe = region;
    safe.x = safe.x;
    safe.y = safe.y;
    safe.width = safe.width;
    safe.height = safe.height;
    safe.x = std::max(0, std::min(safe.x, imageSize.width - 1));
    safe.y = std::max(0, std::min(safe.y, imageSize.height - 1));
    safe.width = std::min(safe.width, imageSize.width - safe.x);
    safe.height = std::min(safe.height, imageSize.height - safe.y);
    return safe;
}

// 实现点位验证方法
vector<LedPoint> validatePointPositions(const vector<LedPoint> &points, float min_valid_neighbors) {
    if (points.size() < 10) return points;

    // 1. 计算连续id点之间的平均间距和标准差
    float total_distance = 0;
    int pair_count = 0;
    vector<float> distances;

    // 记录连续id点对
    vector<pair<const LedPoint *, const LedPoint *>> consecutive_pairs;
    for (size_t i = 0; i < points.size(); i++) {
        for (size_t j = i + 1; j < points.size(); j++) {
            if (abs(points[i].id - points[j].id) == 1) {
                float dist = norm(points[i].position - points[j].position);
                total_distance += dist;
                distances.push_back(dist);
                pair_count++;
                consecutive_pairs.push_back({&points[i], &points[j]});
            }
        }
    }

    if (pair_count == 0) return points;

    float avg_distance = total_distance / pair_count;

    // 计算标准差
    float variance = 0;
    for (float dist: distances) {
        float diff = dist - avg_distance;
        variance += diff * diff;
    }
    float std_dev = sqrt(variance / pair_count);
    float max_allowed_deviation = avg_distance + 2.5 * std_dev;
    float max_allowed_deviation_reliable = avg_distance + 2 * std_dev;
    float max_allowed_deviation_maxIc = avg_distance + 5 * std_dev;

    LOGD(LOG_TAG, "点位间距统计 - 平均: %.2f, 标准差: %.2f, 最大允许偏差: %.2f",
         avg_distance, std_dev, max_allowed_deviation);

    // 2. 找出可信的连续点序列
    unordered_set<int> reliable_ids;
    for (const auto &[p1, p2]: consecutive_pairs) {
        float dist = norm(p1->position - p2->position);
        if (abs(dist - avg_distance) <= max_allowed_deviation_reliable) {
            reliable_ids.insert(p1->id);
            reliable_ids.insert(p2->id);
        }
    }

    LOGD(LOG_TAG, "找到可信的点数量: %zu", reliable_ids.size());

    // 3. 验证每个点的合理性
    vector<LedPoint> validPoints;
    for (const auto &point: points) {
        // 如果点已经在可信序列中，直接保留
        if (reliable_ids.count(point.id) > 0) {
            validPoints.push_back(point);
            continue;
        }

        // 否则，只需要检查与最近id的可信点的距离
        vector<pair<int, const LedPoint *>> nearby_reliable;
        for (const auto &other: points) {
            if (reliable_ids.count(other.id) > 0) {
                nearby_reliable.push_back({abs(point.id - other.id), &other});
            }
        }

        if (nearby_reliable.empty()) {
            // 如果没有可信点，使用原来的方法验证
            vector<pair<float, const LedPoint *>> nearest_points;
            for (const auto &other: points) {
                if (point.id != other.id) {
                    int id_diff = abs(point.id - other.id);
                    nearest_points.push_back({id_diff, &other});
                }
            }

            sort(nearest_points.begin(), nearest_points.end(),
                 [](const auto &a, const auto &b) { return a.first < b.first; });

            if (nearest_points.size() > 4) {
                nearest_points.resize(4);
            }

            int valid_neighbors = 0;
            for (const auto &[id_diff, near_point]: nearest_points) {
                float dist = norm(point.position - near_point->position);
                float expected_dist = avg_distance * id_diff;
                if (abs(dist - expected_dist) <= max_allowed_deviation * id_diff) {
                    valid_neighbors++;
                }
            }

            if (valid_neighbors >= min_valid_neighbors) {
                validPoints.push_back(point);
//                LOGD(LOG_TAG, "通过邻近点验证保留点 id=%d (%.1f, %.1f), 有效邻居: %d",
//                     point.id, point.position.x, point.position.y, valid_neighbors);
            } else {
                LOGW(LOG_TAG, "2-丢弃异常点 id=%d (%.1f, %.1f), 有效邻居: %d",
                     point.id, point.position.x, point.position.y, valid_neighbors);
            }
        } else {
            // 只检查与最近的可信点的距离
            sort(nearby_reliable.begin(), nearby_reliable.end(),
                 [](const auto &a, const auto &b) { return a.first < b.first; });

            const auto &[id_diff, nearest] = nearby_reliable[0];
            float dist = norm(point.position - nearest->position);
            float expected_dist = avg_distance * id_diff;

            if (point.id > g_colorSplitter.getIcNum() - 10 &&
                abs(dist - expected_dist) <= max_allowed_deviation_maxIc * id_diff) {
                validPoints.push_back(point);
//                LOGD(LOG_TAG,
//                     "通过可信点验证保留点 id=%d (%.1f, %.1f), 参考点id=%d, 距离偏差: %.2f",
//                     point.id, point.position.x, point.position.y,
//                     nearest->id, abs(dist - expected_dist));
            } else if (abs(dist - expected_dist) <= max_allowed_deviation_reliable * id_diff) {
                validPoints.push_back(point);
//                LOGD(LOG_TAG,
//                     "通过可信点验证保留点 id=%d (%.1f, %.1f), 参考点id=%d, 距离偏差: %.2f",
//                     point.id, point.position.x, point.position.y,
//                     nearest->id, abs(dist - expected_dist));
            } else {
                LOGW(LOG_TAG, "丢弃异常点 id=%d (%.1f, %.1f), 与参考点(id=%d)距离偏差过大: %.2f",
                     point.id, point.position.x, point.position.y,
                     nearest->id, abs(dist - expected_dist));
            }
        }
    }

    LOGW(LOG_TAG, "点位验证 - 原始点数: %zu, 有效点数: %zu (其中可信点: %zu)",
         points.size(), validPoints.size(), reliable_ids.size());

    return validPoints;
}
