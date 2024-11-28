#include "features.hpp"
#include "sequence.hpp"
#include "anomaly.cpp"
#include "select.hpp"
#include "interpolate682x.hpp"
#include "interpolate70cx.hpp"
#include "inferredp.hpp"
#include "aligner2.cpp"
#include "EnhancedLookup.cpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

#include <setjmp.h>
#include <iomanip>
#include <sstream>
#include <csetjmp>

/**
 * 得分计算序列
 */
vector<vector<int>> scoreVV = {{0, 1,   2},
                               {0, 3,   5},
                               {0, 6,   10},
                               {0, 11,  19},
                               {0, 20,  36},
                               {0, 37,  69},
                               {0, 70,  134},
                               {0, 135, 263},
                               {0, 264, 520}};

//绿 红 绿 红 绿 绿 绿 绿 绿
//2+3+10+11+36+69+134+263+520
int lightType = 0;
//得分点集合
vector<LightPoint> pPoints;
vector<Point2f> pPointXys;
unordered_map<int, vector<LightPoint>> pointsStepMap;
//记录有效帧
unordered_map<int, Mat> frameStepMap;
/**
* 得分错点集合
*/
vector<LightPoint> errorSerialVector;
unordered_map<int, vector<LightPoint>> sequenceTypeMap;

EnhancedChristmasTreeAligner aligner;

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat, vector<Mat> &outMats) {
    Mat srcResize, alignMat;
    // 指定缩放后的尺寸
    Size newSize(640, 640);
    if (frameStep > STEP_VALID_FRAME_START) {
        alignMat = aligner.alignImage(frameStepMap[STEP_VALID_FRAME_START], originalMat, outMats);
        if (alignMat.empty()) {
            return alignMat;
        }
        LOGD(LOG_TAG, "frameStep=%d, alignResize %d-%d", frameStep, alignMat.rows, alignMat.cols);
        resize(alignMat, srcResize, newSize);
        frameStepMap[frameStep] = alignMat;
    } else {
        release();
        aligner = EnhancedChristmasTreeAligner();
        resize(originalMat, srcResize, newSize);
        frameStepMap[frameStep] = originalMat;
    }
    return srcResize;
}

void releaseFrameStepMap(unordered_map<int, cv::Mat> &map) {
    for (auto &pair: map) {
        pair.second.release();  // 显式释放每个 Mat 对象
    }
    map.clear();  // 清空 map
}

/**释放资源*/
void release() {
    try {
        pointsStepMap.clear();
        releaseFrameStepMap(frameStepMap);
        pPointXys.clear();
        pPointXys.shrink_to_fit();
        pPoints.clear();
        pPoints.shrink_to_fit();
        sequenceTypeMap.clear();
        errorSerialVector.clear();
        errorSerialVector.shrink_to_fit();
    } catch (...) {
        LOGE(LOG_TAG, "error------release");
    }
}


/**
 * 根据定义好的步骤进行灯带排序
 * @param frameStep 当前轮数
 * @param resultObjects 当前tf识别的结果以及opencv找色的结果
 * @param radiusCircle 当前绘制测试图的点位半径
 * @param outMats 输出流程中的测试图像
 * @return
 */
String
sortStripByStep(int frameStep, vector<LightPoint> &resultObjects, int lightTypeP,
                vector<Mat> &outMats) {
    lightType = lightTypeP;
    try {
        if (frameStep == STEP_VALID_FRAME_START) {
            Mat src = frameStepMap[frameStep].clone();
            if (!pPointXys.empty())pPointXys.clear();
            for (const auto &curPoint: resultObjects) {
                pPointXys.push_back(curPoint.position);
            }

            Mat image = frameStepMap[frameStep].clone();
            try {
                // 创建检测器实例
                LEDDetector detector(image, pPointXys);
                // 分析已知灯珠特征
                detector.analyzeKnownLEDs();
                // 查找遗漏的灯珠
                vector<Point2f> missing_leds = detector.findMissingLEDs();
                LOGW(LOG_TAG, " 查找遗漏的灯珠 missing_leds = %d KnownPositions = %d",
                     missing_leds.size(), detector.getKnownPositions().size());
//            Mat result = detector.visualize();
//            outMats.push_back(result);
                pPointXys.clear();
                pPoints.clear();
                for (const auto &item: detector.getKnownPositions()) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPoints.push_back(lp);
                    pPointXys.push_back(item);
                }
                for (const auto &item: missing_leds) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPoints.push_back(lp);
                    pPointXys.push_back(item);

                }
            } catch (...) {
                LOGE(LOG_TAG, "error LEDDetector");
                for (const auto &item: pPointXys) {
                    LightPoint lp = LightPoint();
                    lp.position = item;
                    pPoints.push_back(lp);
                    pPointXys.push_back(item);
                }
            }
            LOGE(LOG_TAG, " pPointXys = %d pPoints = %d", pPointXys.size(), pPoints.size());
        }
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常1");
        return "{}";
    }

    //定位特征点
    vector<LightPoint> findVector = findColorType(frameStepMap[frameStep], frameStep, pPoints,
                                                  outMats);
    pointsStepMap[frameStep] = findVector;
    LOGD(LOG_TAG, "pointsStepMap frameStep=%d pointsStepMap=%d   测试图 =%d", frameStep,
         pointsStepMap.size(), outMats.size());
    if (pointsStepMap.size() == getMaxStepCnt()) {
        //--------------------------------------- 开始识别 ---------------------------------------
        vector<Point2f> trapezoid4Points;
        LampBeadsProcessor processor = sortLampBeads(frameStepMap[STEP_VALID_FRAME_START], outMats,
                                                     trapezoid4Points);
        release();
        //回调函数
        return splicedJson(lightPointsToJson(processor.totalPoints, lightTypeP),
                           point2iToJson(trapezoid4Points));
    }
    return "";
}

void drawPointsMatOut(const Mat &src, const vector<LightPoint> lightPoints, vector<Mat> &outMats) {
    try {
        Mat dstCircle = src.clone();
        for (auto &pPoint: pPoints) {
            try {
                LightPoint lpoint = pPoint;
                Point2f center = lpoint.position;
                if (center.x < 8 || center.x >= src.cols - 8 || center.y < 8 ||
                    center.y >= src.rows - 8) {
                    continue;
                }
                buildRect(pPoint, src, 7);
                if (pPoint.errorStatus != NORMAL) {
                    circle(dstCircle, lpoint.position, 7, Scalar(255, 255, 0, 150), 2);
                } else {
                    circle(dstCircle, lpoint.position, 7, Scalar(0, 0, 0, 150), 2);
                }
            } catch (...) {}
        }
        for (const auto &lightPoint: lightPoints) {
            try {
                LightPoint lpoint = lightPoint;
                Point2f center = lpoint.position;
                if (center.x < 8 || center.x >= src.cols - 8 || center.y < 8 ||
                    center.y >= src.rows - 8) {
                    continue;
                }
                center.x = static_cast<int>(center.x);
                center.y = static_cast<int>(center.y);
                circle(dstCircle, center, 7, Scalar(0, 255, 255, 150), 2);
                putText(dstCircle, to_string(lightPoint.label), center,
                        FONT_HERSHEY_SIMPLEX,
                        0.6,
                        Scalar(0, 0, 255),
                        1);
            } catch (...) {}
        }
        outMats.push_back(dstCircle);
    } catch (std::exception &e) {
        LOGE(LOG_TAG, "drawPointsMatOut  e =  %s", e.what());
    } catch (...) {
        LOGE(LOG_TAG, "drawPointsMatOut error");
    }
}

void remove4AnomalyDetector(LampBeadsProcessor &processor) {
    if (getIcNum() >= 500 && processor.totalPoints.size() > 20) {
        //对补全的点进行排序
        sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);
        float distanceThreshold = 6.0f;  // 距离阈值系数
        int labelDiffThreshold = 10;   // 允许的最大标签差
        int densityNeighbors = 10;  // 用于计算密度的邻居数
        AnomalyDetector detector(processor.totalPoints, processor.averageDistance,
                                 distanceThreshold,
                                 labelDiffThreshold, getIcNum(), densityNeighbors);
        vector<int> anomalies = detector.detectAnomalies();
        LOGE(TAG_DELETE, "remove4AnomalyDetector errorPoints=%d", anomalies.size());
        for (int i = anomalies.size() - 1; i >= 0; i--) {
            int pointIndex = anomalies[i];
            if (pointIndex >= processor.totalPoints.size() || pointIndex < 0) {
                LOGE(TAG_DELETE, "remove4AnomalyDetector 擦除脏数据失败");
                continue;
            }
            LOGD(TAG_DELETE,
                 "remove4AnomalyDetector erase index=%d, label=%d  errorPoint = %f x %f",
                 anomalies[i],
                 processor.totalPoints[anomalies[i]].label,
                 processor.totalPoints[anomalies[i]].position.x,
                 processor.totalPoints[anomalies[i]].position.y);
            processor.totalPoints.erase(processor.totalPoints.begin() + anomalies[i]);
        }
    }
}

/**
 * 对灯带光点排序
 */
LampBeadsProcessor
sortLampBeads(Mat &src, vector<Mat> &outMats, vector<Point2f> &trapezoid4Points) {
    int scoreMin = getScoreMin();
    int scoreMax = getScoreMax();
    int maxFrameStep = getMaxStepCnt();

    LampBeadsProcessor processor = LampBeadsProcessor(scoreMin, scoreMax, maxFrameStep);
    if (pPoints.empty())return processor;

    LOGW(LOG_TAG, "sortLampBeads pPoints=%d   scoreMin=%d , scoreMax = %d ,endStep = %d",
         pPoints.size(), scoreMin, scoreMax, maxFrameStep);

    vector<int> sameColorScore = getSameColorVector();

    /*统计得分*/
    int ret = statisticalScorePoints(src, outMats, processor);

    if (ret == 0) {
        LOGE(LOG_TAG, "统计得分失败");
        return processor;
    }

    if (processor.normalPoints.empty())return processor;

    //计算点位平均距离
    double averageDistance = calculateAverageDistance(processor);
    drawPointsMatOut(src, processor.normalPoints, outMats);//todo:linpeng

    if (processor.totalPoints.size() > 20) {
        detectOutlierPoints(processor.totalPoints, errorSerialVector, averageDistance, 8);
    }

    /*推测中间夹点*/
    processor.totalPoints = decisionCenterPoints2(processor.normalPoints, averageDistance);

    remove4AnomalyDetector(processor);

    if (processor.totalPoints.size() < 4)return processor;

    LOGD(LOG_TAG, "processor.totalPoints = %d", processor.totalPoints.size());

    decisionRightLeftPoints(processor.totalPoints, false);

    /*处理分值相同的点*/
    processSamePoints(src, outMats, processor.totalPoints, errorSerialVector, averageDistance,
                      processor.sameSerialNumMap);

    processor.totalPoints = fillMissingPoints(processor.totalPoints, averageDistance);

    if (processor.totalPoints.size() > 20) {
        detectOutlierPoints(processor.totalPoints, errorSerialVector, averageDistance);
    }

    if (processor.totalPoints.size() > 50) {
        //对补全的点进行排序
        sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);

        float distanceThreshold = 5.0f;  // 距离阈值系数
        int labelDiffThreshold = 25;   // 允许的最大标签差
        if (getIcNum() == 500) {
            labelDiffThreshold = 25;
        } else if (getIcNum() == 200) {
            labelDiffThreshold = 35;
        }
        int densityNeighbors = 20;  // 用于计算密度的邻居数
        AnomalyDetector detector(processor.totalPoints, processor.averageDistance,
                                 distanceThreshold,
                                 labelDiffThreshold, getIcNum(), densityNeighbors);
        vector<int> anomalies = detector.detectAnomalies();
        LOGE(TAG_DELETE, "errorPoints=%d", anomalies.size());
        for (int i = anomalies.size() - 1; i >= 0; i--) {
            int pointIndex = anomalies[i];
            if (pointIndex >= processor.totalPoints.size() || pointIndex < 0) {
                LOGE(TAG_DELETE, "擦除脏数据失败");
                continue;
            }
            LOGD(TAG_DELETE, "erase index=%d, label=%d  errorPoint = %f x %f",
                 anomalies[i],
                 processor.totalPoints[anomalies[i]].label,
                 processor.totalPoints[anomalies[i]].position.x,
                 processor.totalPoints[anomalies[i]].position.y);
            processor.totalPoints.erase(processor.totalPoints.begin() + anomalies[i]);
        }
        int sizeOld = processor.totalPoints.size();
        decisionRightLeftPoints(processor.totalPoints, true);
        LOGD(LOG_TAG, "decisionRightLeftPoints add = %d", processor.totalPoints.size() - sizeOld);
    }

    //处理剩余无序点位
    decisionRemainingPoints(processor);

    /*删除离群点+构建梯形*/
    if (lightType == TYPE_H70CX_3D) {
        Mat trapezoidMat = src.clone();
        vector<Point2f> point4Trapezoid;
        for (auto &totalPoint: processor.totalPoints) {
            point4Trapezoid.push_back(totalPoint.position);
        }
        LOGD(LOG_TAG, "point4Trapezoid = %d", point4Trapezoid.size());
        ret = getMinTrapezoid(trapezoidMat, point4Trapezoid, trapezoid4Points);
        if (ret != 1) {
            LOGE(LOG_TAG, "构建梯形异常");
        }
        trapezoidMat.release(); //todo: linpeng
//        outMats.push_back(trapezoidMat);
    } else {
        try {
            processor.totalPoints = interpolateAndExtrapolatePoints(processor.totalPoints,
                                                                    getIcNum());
        } catch (...) {
            LOGE(LOG_TAG, "interpolateAndExtrapolatePoints error");
        }
    }
    //todo: linpeng
//    drawPointsMatOut(src, processor.totalPoints, outMats);
    return processor;
}


bool reCheckScore(LampBeadsProcessor &processor, std::vector<LightPoint> &lightPoints) {
    LightPoint result = {EMPTY_POINT};
    int sameSize = lightPoints.size();
    int scoreMin = processor.scoreMin;
    int scoreMax = processor.scoreMax;
    int maxFrameStep = processor.maxFrameStep;
    vector<int> sameColorScore = getSameColorVector();
    for (int i = 0; i < lightPoints.size(); /* 手动控制索引 */) {
        LightPoint &item = lightPoints[i];
        int score = 0;

        // 计算新的 score
        for (int step = 0; step < getMaxStepCnt(); ++step) {
            cv::Mat e;
            vector<vector<Point>> contours;
            LightPoint lightPoint = meanLightColor(frameStepMap[step], contours, step, item, e, 4);
            score += scoreVV[step][lightPoint.type];
        }
        ++i;
        if (item.score != score) {
            item.score = score;
            sameSize--;
            if (score < scoreMin) {
                continue;
            }
            if (score > scoreMax) {
                continue;
            }
            if (score == sameColorScore[0]) {
                sequenceTypeMap[0].push_back(pPoints[i]);
                continue;
            }
            if (score == sameColorScore[1]) {
                sequenceTypeMap[1].push_back(pPoints[i]);
                continue;
            }
            // 处理getScoreMax() - n的分数
            for (int n = 1; n <= 10; ++n) { // 假设你想处理到getScoreMax() - 7
                int currentScore = getScoreMax() - n;
                if (score == currentScore) {
                    sequenceTypeMap[n + 1].push_back(pPoints[i]);
                    continue;
                }
            }
            syncLightIndex(item, score, lightType);
            LOGD(LOG_TAG, "重新测量后的label值：%d", item.label);
            // 检查 processor.normalPoints 中是否已有同样 score 的点
            bool hasContainer = std::any_of(processor.normalPoints.begin(),
                                            processor.normalPoints.end(),
                                            [&item](const LightPoint &point) {
                                                return point.score == item.score;
                                            });

            if (!hasContainer && item.label > 0) {
                LOGW(LOG_TAG, "在normalPoints插入新值：%d   %f - %f", item.label, item.position.x,
                     item.position.y);
                processor.normalPoints.push_back(item);
            }
            --i;
            lightPoints.erase(lightPoints.begin() + i);
        } else {
            result = item;
        }
    }
    if (lightPoints.size() == 1 && result.errorStatus != EMPTY_POINT) {
        syncLightIndex(result, result.score, lightType);
        LOGW(LOG_TAG, "最终纠正值：%d   %f - %f", result.label, result.position.x,
             result.position.y);
        processor.normalPoints.push_back(result);
        return true;
    } else if (lightPoints.empty()) {
        return true;
    }
    return false;
}


/**
 * 统计所有得分
 */
int statisticalScorePoints(Mat &src, vector<Mat> &outMats, LampBeadsProcessor &processor) {
    int scoreMin = processor.scoreMin;
    int scoreMax = processor.scoreMax;
    int maxFrameStep = processor.maxFrameStep;
    vector<int> sameColorScore = getSameColorVector();
    //消除
    vector<int> eraseVector = polyPoints(pPointXys, 3, 2.3);
    sort(eraseVector.begin(), eraseVector.end(), std::greater<int>());
    LOGD(LOG_TAG, "消除点集合 eraseVector=%d  scoreMin=%d  scoreMax=%d, sameColor=%d - %d",
         eraseVector.size(),
         scoreMin, scoreMax, sameColorScore[0], sameColorScore[1]);

    for (int n = 1; n <= 10; ++n) { // 假设你想处理到getScoreMax() - 7
        int currentScore = scoreMax - n;
    }

    for (int index: eraseVector) {
        auto erasePoint = pPoints.begin() + index;
        erasePoint->errorStatus = ERASE_POINT;
    }
    sequenceTypeMap.clear();
    for (int i = 0; i < 12; i++) {
        sequenceTypeMap[i] = vector<LightPoint>();
    }
    // 1. 统计分值相同的index
    // 2. 标记异常分点
    int normalIndex = 0;
    /*对得分进行排序*/
    sort(pPoints.begin(), pPoints.end(), compareScore);
    for (int i = 0; i < pPoints.size(); i++) {
        //--------------------绘制--------------------
        Point2f center = pPoints[i].position;
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);

        processor.pointXys.push_back(center);

        int score = 0;
        for (int step = STEP_VALID_FRAME_START; step < maxFrameStep; step++) {
            if (pointsStepMap[step].size() != pPoints.size()) {
                LOGE(LOG_TAG, "pointsStepMap[step] size error");
                return 0;
            }
            score += scoreVV[step][pointsStepMap[step][i].type];
        }
        pPoints[i].score = score;

        if (pPoints[i].errorStatus == ERASE_POINT) {
            errorSerialVector.push_back(pPoints[i]);
            continue;
        }
        if (score < scoreMin) {
//            LOGV(LOG_TAG, "异常分值(<scoreMin=%d)，endStep = %d，i = %d，index = %d", scoreMin,
//                 maxFrameStep, i,
//                 (score - scoreMin));
            continue;
        }
        if (score > scoreMax) {
            LOGV(LOG_TAG, "异常分值(>scoreMax=%d)，i=%d，index=%d", scoreMax, i, (score - scoreMin));
            continue;
        }
        if (score == sameColorScore[0]) {
            sequenceTypeMap[0].push_back(pPoints[i]);
            continue;
        }
        if (score == sameColorScore[1]) {
            sequenceTypeMap[1].push_back(pPoints[i]);
            continue;
        }
        bool hasCache = false;
        // 处理getScoreMax() - n的分数
        for (int n = 1; n <= 10; ++n) { // 假设你想处理到getScoreMax() - 7
            int currentScore = getScoreMax() - n;
            if (score == currentScore) {
                sequenceTypeMap[n + 1].push_back(pPoints[i]);
                hasCache = true;
                break;
            }
        }
        if (hasCache) {
            continue;
        }
        //计算序列
        syncLightIndex(pPoints[i], score, lightType);

        if (pPoints[i].label < 0 || pPoints[i].label > getIcNum()) {
            errorSerialVector.push_back(pPoints[i]);
        } else if ((lightType == TYPE_H70CX_3D || lightType == TYPE_H70CX_2D) &&
                   pPoints[i].label % 2 != 0) {
            LOGW(LOG_TAG, "error label: %d", pPoints[i].label);
            //70dx的序列是奇数点位，不满足的话就是推错点了
            errorSerialVector.push_back(pPoints[i]);
        } else {
            if (pPoints[i].position.x == 0.0 && pPoints[i].position.y == 0.0) {
                LOGE(LOG_TAG, "=================sameSerialNumMap 异常点");
            }
            processor.sameSerialNumMap[pPoints[i].label].push_back(pPoints[i]);
            normalIndex++;
        }
    }
    int samePointsSize = 0;
    for (const auto &entry: processor.sameSerialNumMap) {
        const vector<LightPoint> &indices = entry.second;
        if (indices.size() == 1) {
            processor.normalPoints.push_back(indices[0]);
        } else {
            samePointsSize++;

        }
    }
//    for (const auto &entry: processor.sameSerialNumMap) {
//        vector<LightPoint> indices = entry.second;
//        if (indices.size() > 1) {
//            reCheckScore(processor, indices);
//        }
//    }

    LOGW(LOG_TAG,
         "normalPoints = %d  重复标签点 = %d  redSameVector = %d  greenSameVector = %d errorSerialVector = %d",
         processor.normalPoints.size(),
         samePointsSize,
         sequenceTypeMap[0].size(),
         sequenceTypeMap[1].size(),
         errorSerialVector.size()
    );
    return 1;
}

/**
 * 计算点位平均距离
 */
double calculateAverageDistance(LampBeadsProcessor &processor) {
    sort(processor.normalPoints.begin(), processor.normalPoints.end(),
         [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

    int averageCnt = 1;
    int diff = 2;
    vector<double> distanceList;
    for (int i = 0; i < processor.normalPoints.size() - 1; i++) {
        LightPoint curLPoint = processor.normalPoints[i];
        LightPoint nextLPoint = processor.normalPoints[i + 1];
        int xx = nextLPoint.label - curLPoint.label;
        if (xx == diff) {
            distanceList.push_back(norm(nextLPoint.position - curLPoint.position));
            averageCnt += 2;
        }
    }
    // 排序
    sort(distanceList.begin(), distanceList.end());

    // 计算要过滤的数量
    size_t numToRemoveMin = distanceList.size() / 5;
    size_t numToRemoveMax = distanceList.size() / 7;

    // 计算初步过滤后的剩余数量
    size_t initialRemainingSize = distanceList.size() - numToRemoveMin - numToRemoveMax;

    // 如果初步过滤后的剩余数量小于或等于5个
    if (initialRemainingSize <= 5) {
        // 先减少过滤的最大值
        while (initialRemainingSize <= 5 && numToRemoveMax > 0) {
            --numToRemoveMax;
            ++initialRemainingSize;
        }
        // 如果仍然不足，减少过滤的最小值
        while (initialRemainingSize <= 5 && numToRemoveMin > 0) {
            --numToRemoveMin;
            ++initialRemainingSize;
        }
        // 如果仍然不足，直接返回错误
        if (initialRemainingSize <= 5) {
            LOGE(LOG_TAG, "Error: Cannot maintain more than 5 elements after filtering.");
            return 65.0;
        }
    }

    // 过滤最小值
    distanceList.erase(distanceList.begin(), distanceList.begin() + numToRemoveMin);

    // 过滤最大值
    distanceList.erase(distanceList.end() - numToRemoveMax, distanceList.end());

    // 计算剩余元素的平均值
    double sum = std::accumulate(distanceList.begin(), distanceList.end(), 0.0);
    double average = sum / (distanceList.size() * 2);
    processor.averageDistance = average;
    LOGW(LOG_TAG, "平均距离 averageDistance = %f  averageCnt = %d", average, averageCnt);
    return average;
}

// 函数：在 inferredList 中寻找符合条件的 LightPoint，并返回找到的点。
// 输入：
// - A 和 C：作为椭圆的定点
// - inferredList：待搜索的 LightPoint 列表
// - averageDistance：用于计算椭圆长轴长度的平均距离
// 输出：
// - 返回找到的符合条件的 LightPoint，并从 inferredList 中删除该点。
LightPoint findAndRemoveClosestInEllipse(const LightPoint &A, const LightPoint &C,
                                         vector<LightPoint> &inferredList,
                                         double averageDistance) {
    // 1. 计算 A 和 C 的距离，得到 AC 的长度
    double AC_distance = cv::norm(A.position - C.position);

    // 2. 确定椭圆长轴的长度，即 AC 距离加上 averageDistance
    double ellipse_major_axis_length = AC_distance + averageDistance;

    // 3. 计算椭圆的中心点，位于 A 和 C 的中点
    cv::Point2f ellipse_center = (A.position + C.position) * 0.5f;

    // 4. 计算从 A 到 C 的单位方向向量，表示椭圆的长轴方向
    cv::Point2f AC_direction = (C.position - A.position) / AC_distance;

    // 用于保存距离中心点最近的符合条件的 LightPoint
    LightPoint closestPoint = {EMPTY_POINT};
    double minDistanceToCenter = std::numeric_limits<double>::max(); // 初始化为最大距离
    auto closestIt = inferredList.end(); // 初始化为无效迭代器

    // 5. 遍历 inferredList 中的每一个点
    for (auto it = inferredList.begin(); it != inferredList.end(); ++it) {
        // 5.1 计算当前点相对于椭圆中心的向量
        Point2f AP_vector = it->position - ellipse_center;

        // 5.2 计算当前点在 AC 方向上的投影长度
        double distance_along_AC = AP_vector.dot(AC_direction);

        // 5.3 计算当前点到 AC 方向的垂直距离
        double distance_perpendicular_AC = norm(AP_vector - distance_along_AC * AC_direction);

        // 6. 确定椭圆的半长轴 a 和半短轴 b
        double a = ellipse_major_axis_length / 2.0; // 半长轴
        double b = a / 2.0;                         // 半短轴，假设短轴为长轴的一半

        // 7. 使用椭圆方程判断当前点是否位于椭圆内部
        // (x/a)^2 + (y/b)^2 <= 1 表示椭圆内部
        if ((distance_along_AC * distance_along_AC) / (a * a) +
            (distance_perpendicular_AC * distance_perpendicular_AC) / (b * b) <= 1) {
            // 7.1 计算当前点到椭圆中心的距离
            double distanceToCenter = cv::norm(it->position - ellipse_center);

            // 7.2 如果该点比之前的点更接近中心点，则更新最近点
            if (distanceToCenter < minDistanceToCenter) {
                minDistanceToCenter = distanceToCenter;
                closestPoint = *it;
                closestIt = it;
            }
        }
    }

    // 8. 如果找到了符合条件的点，则从 inferredList 中删除该点
    if (closestIt != inferredList.end()) {
        inferredList.erase(closestIt);
    }

    // 返回找到的距离中心最近的点
    return closestPoint;
}

// 函数：在 inferredList 中寻找满足条件的 LightPoint，并返回找到的点。
// 输入：
// - A 和 C：定义圆心的两个点
// - inferredList：待搜索的 LightPoint 列表
// - averageDistance：作为搜索圆的半径
// 输出：
// - 返回找到的符合条件的 LightPoint，并从 inferredList 中删除该点。
LightPoint findAndRemoveClosestInCircle(const LightPoint &A, const LightPoint &C,
                                        std::vector<LightPoint> &inferredList,
                                        double averageDistance) {
    // 1. 计算圆心（A 和 C 的中点）
    cv::Point2f circle_center = (A.position + C.position) * 0.5f;

    // 2. 设置搜索半径，即输入参数 averageDistance
    double radius = averageDistance;

    // 用于保存距离中心点最近的符合条件的 LightPoint
    LightPoint closestPoint;
    double minDistanceToCenter = std::numeric_limits<double>::max(); // 初始化为最大值
    auto closestIt = inferredList.end(); // 初始化为无效迭代器

    // 3. 遍历 inferredList 中的每一个点
    for (auto it = inferredList.begin(); it != inferredList.end(); ++it) {
        // 3.1 计算当前点到圆心的距离
        double distanceToCenter = cv::norm(it->position - circle_center);

        // 3.2 检查该点是否在圆内
        if (distanceToCenter <= radius) {
            // 如果该点比之前的点更接近圆心，则更新最近点
            if (distanceToCenter < minDistanceToCenter) {
                minDistanceToCenter = distanceToCenter;
                closestPoint = *it;
                closestIt = it;
            }
        }
    }

    // 4. 如果找到了符合条件的点，则从 inferredList 中删除该点
    if (closestIt != inferredList.end()) {
        inferredList.erase(closestIt);
    }

    // 返回找到的距离中心最近的点
    return closestPoint;
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

/**
 * 推测中间夹点
 */
vector<LightPoint>
decisionCenterPoints2(const vector<LightPoint> &input, double averageDistance) {
    //补充不连续段落,记录 last 临时存储 -1的原因是为了补0
    LOGD(LOG_TAG, "=============推测中间夹点New input=%d=============", input.size());
    vector<LightPoint> points = input;
    sort(points.begin(), points.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });
    logLabels(input);
    int totalDiff_1 = 0;
    int totalDiff_4 = 0;
    int totalAdd_4 = 0;
    vector<LightPoint> inferredResult;
    for (int i = 1; i < points.size(); ++i) {
        LightPoint startLPoint = points[i - 1];
        LightPoint endLPoint = points[i];
        Point2f startPoint = startLPoint.position;
        Point2f endPoint = endLPoint.position;
        int missingCount = endLPoint.label - startLPoint.label;
        if (missingCount == 2) {
            if (norm(startPoint - endPoint) > averageDistance * 4) {
                continue;
            }
            totalDiff_1++;
            int inferredLightIndex = startLPoint.label + 1;
            if (inferredLightIndex > getIcNum()) continue;
            // 处理中间缺失一个点的情况
            int sequenceType = getNonSequenceType(inferredLightIndex, lightType);

            if (sequenceType == -1) {
                LOGD(TAG_INFERRED, "new 非推断序号 = %d", inferredLightIndex);
                continue;
            }
            vector<LightPoint> &inferredList = sequenceTypeMap[sequenceType];

            for (const auto &item: inferredList) {
                if (item.position.x == 0) {
                    LOGE(TAG_INFERRED, "2---inferredList %f -%f", item.position.x, item.position.y);
                }
            }

            LightPoint findLp = findAndRemoveClosestInEllipse(endLPoint, startLPoint, inferredList,
                                                              averageDistance * 1.5);

            findLp.label = inferredLightIndex;
            if (findLp.errorStatus == EMPTY_POINT) {
                Point2f ellipse_center = (endPoint + startPoint) * 0.5f;
                findLp.position = ellipse_center;
                findLp.errorStatus = NORMAL;
            }
            inferredResult.push_back(findLp);
        }
    }
    inferredResult.insert(inferredResult.end(), input.begin(), input.end());
    LOGD(LOG_TAG,
         "inferredResult = %d normalPoints = %d  totalDiff_1 = %d totalDiff_4 = %d",
         inferredResult.size(),
         input.size(), totalDiff_1, totalDiff_4);
    return inferredResult;
}


/**
 * 推测中间夹点
 */
vector<LightPoint>
decisionCenterPoints(vector<LightPoint> &input, double averageDistance) {
    //补充不连续段落,记录 last 临时存储 -1的原因是为了补0
    LOGD(LOG_TAG, "推测中间夹点");
    vector<LightPoint> result;
    int expectedLabel = 0;
    vector<LightPoint> points = input;
    sort(points.begin(), points.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });
    for (int i = 1; i < points.size(); ++i) {
        if (expectedLabel < points[i].label) {
            int missingCount = points[i].label - expectedLabel;
            LightPoint startLPoint = (i == 0) ? points[i] : points[i - 1];
            LightPoint endLPoint = points[i];
            Point2f startPoint = startLPoint.position;
            Point2f endPoint = endLPoint.position;

            if (missingCount == 1 && i > 0) {
                // 处理中间缺失一个点的情况
                LightPoint interpolated = inferredCenter(averageDistance, endLPoint, startLPoint,
                                                         true);
                interpolated.label = expectedLabel;
                if (interpolated.errorStatus != EMPTY_POINT && interpolated.label < getIcNum() &&
                    interpolated.label >= 0) {
                    result.push_back(interpolated);
                } else {
                    //当点位的前面或者后面是连续的点时，自动补全中间这个点
                    try {
                        if (canBelievedAB(startPoint, endPoint, points, i, averageDistance)) {
                            LightPoint lp = LightPoint(expectedLabel);
                            cv::Point2f midpoint = Point2f((startPoint.x + endPoint.x) / 2,
                                                           (startPoint.y + endPoint.y) / 2);
                            lp.position = midpoint;
                            lp.label = expectedLabel;
                            result.push_back(lp);
                            LOGD(LOG_TAG,
                                 "canBelievedAB expectedLabel = %d  label=%d  midpoint= %f - %f",
                                 expectedLabel, lp.label, lp.position.x, lp.position.y);
                        }
                    } catch (...) {
                        LOGE(LOG_TAG, "decisionCenterPoints error!");
                    }
                }
            }
        }
        result.push_back(points[i]);
        expectedLabel = points[i].label + 1;
    }
    LOGD(LOG_TAG, "normalPoints = %d totalPoints = %d", input.size(),
         result.size());
    return result;
}


/**
 * 从红绿固定点和错点中推测点位
 */
void decisionRightLeftPoints(vector<LightPoint> &totalPoints, bool findErrorPoints) {
    if (totalPoints.size() < 4) return;
    sort(totalPoints.begin(), totalPoints.end(), [](const LightPoint &a, const LightPoint &b) {
        return a.label < b.label;
    });

    try {
        bool enable4BeginLeft = true;//起点往前补点
        for (auto it = totalPoints.begin(); it < totalPoints.end(); ++it) {
            auto beginLP = totalPoints.begin();
            auto endLP = totalPoints.end();
            if (it == beginLP) {
                LightPoint curLPoint = totalPoints[0];
                LightPoint nextLPoint = totalPoints[1];
                int inferredNextDiff = nextLPoint.label - curLPoint.label;
                //第一个点
                if (it->label > 1 && enable4BeginLeft) {
                    LOGD(LOG_TAG, "第1个点之前缺失，begin : %d", it->label);
                    LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint,
                                                               findErrorPoints);
                    if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.label >= 0 &&
                        inferredPoint.label < getIcNum()) {
                        totalPoints.insert(totalPoints.begin(),
                                           inferredPoint);
                        LOGD(LOG_TAG, "补充点位 = %d  ===》重新遍历，直到往前补点失败",
                             inferredPoint.label);
                        it--;
                        continue;
                    } else {
                        enable4BeginLeft = false;
                    }
                }
                if (inferredNextDiff > 1 && totalPoints.size() >= 3) {
                    LOGD(LOG_TAG, "第1个点之后有缺失，begin : %d", it->label);
                    LightPoint nextNextLPoint = totalPoints[2];
                    bool abcHorizontal = isApproximatelyHorizontal(curLPoint.position,
                                                                   nextLPoint.position,
                                                                   nextNextLPoint.position);
                    if (abcHorizontal) {
                        while (inferredNextDiff > 1) {
                            LightPoint inferredPoint = inferredAB2Next(nextNextLPoint, nextLPoint,
                                                                       findErrorPoints);
                            if (inferredPoint.errorStatus != EMPTY_POINT &&
                                inferredPoint.label >= 0 &&
                                inferredPoint.label < getIcNum()) {

                                LOGD(LOG_TAG, "补充点位 = %d", inferredPoint.label);
                                //从next（i+1）的前一个插入
                                totalPoints.insert(totalPoints.begin() + 1,
                                                   inferredPoint);
                                if (totalPoints[2].label !=
                                    inferredPoint.label) {
                                    LOGE(LOG_TAG, "3-----------插入错误");
                                }
                                nextLPoint = inferredPoint;
                                inferredNextDiff--;
                            } else {
                                inferredNextDiff = 0;
                            }
                        }
                    }
                }
            } else if (it == endLP - 1 || it == endLP) {
                //倒数二个点
            } else {
                int i = distance(totalPoints.begin(), it);
                LightPoint curLPoint = totalPoints[i];
                LightPoint nextLPoint = totalPoints[i + 1];
                LightPoint lastLPoint = totalPoints[i - 1];
                //计算下一个点是否缺失
                int inferredNextDiff = nextLPoint.label - curLPoint.label;
                bool inferred2Right = true;
                //代表nextPoint角标
                int nextRightAdd = 0;
                int lastLeftAdd = 0;
                while (inferredNextDiff > 1) {
                    LightPoint inferredPoint;
                    if (inferred2Right) {
                        //首先往右边推断
                        inferredPoint = inferredRight(curLPoint, lastLPoint, nextLPoint, i,
                                                      totalPoints, findErrorPoints);
                        if (inferredPoint.errorStatus != EMPTY_POINT &&
                            inferredPoint.label >= 0 &&
                            inferredPoint.label < getIcNum()) {
                            lastLPoint = curLPoint;
                            curLPoint = inferredPoint;
                            inferredNextDiff--;
                            //往右边挪动一位
                            nextRightAdd++;
                        } else {//往右边插入值不满足时,开始从下一个节点往前插入值
                            inferred2Right = false;
                        }
                    } else {
                        //从最右边开始往左边补点
                        int index = i + 1 + nextRightAdd;
                        if (index > totalPoints.size() - 2 || index < 1) {
                            inferredNextDiff = 0;
                            continue;
                        }
                        curLPoint = totalPoints[index];
                        if (curLPoint.label >= getIcNum()) {
                            inferredNextDiff = 0;
                            continue;
                        }
                        nextLPoint = totalPoints[index + 1];
                        lastLPoint = totalPoints[index - 1];

                        LOGD(LOG_TAG,
                             "向左推断，当前点序号=%d, index = %d, lastLPoint = %d, nextLPoint = %d",
                             curLPoint.label, index, lastLPoint.label,
                             nextLPoint.label);

                        inferredPoint = inferredLeft(curLPoint, lastLPoint, nextLPoint, index,
                                                     totalPoints, findErrorPoints);
                        if (inferredPoint.errorStatus != EMPTY_POINT) {
                            lastLPoint = curLPoint;
                            curLPoint = inferredPoint;
                            inferredNextDiff--;
                            lastLeftAdd++;
                        } else {
                            inferredNextDiff = 0;
                        }
                    }
                }
                int offset = nextRightAdd + lastLeftAdd;
                auto newPosition = std::next(it, offset);
                if (newPosition >= totalPoints.end()) {
                    break;
                }
                it = newPosition;
                // 注意边界判断
                int dis = totalPoints.end() - it - 1;
                if (dis >= totalPoints.size()) {
                    break;
                }
                // 添加安全检查
                if (it >= totalPoints.end()) {
                    break;
                }
                // 使用安全的方式访问 label
                try {
                    if (it->label > getIcNum()) {
                        break;
                    }
                } catch (const std::exception &e) {
                    LOGE(LOG_TAG, "访问 label 时发生异常: %s", e.what());
                    break;
                }
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "异常状态13");
    }
}

vector<LightPoint> fillMissingPoints(const vector<LightPoint> &totalPoints, double avgDistance) {
    try {
        vector<LightPoint> result = totalPoints;
        // 按label排序totalPoints
        sort(result.begin(), result.end(),
             [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

        // 定义一个函数来计算两点之间的距离
        auto distance = [](const Point2f &p1, const Point2f &p2) {
            return norm(p1 - p2);
        };

        // 遍历排序后的点集
        for (size_t i = 1; i < result.size() - 1; ++i) {
            int prev_label = result[i - 1].label;
            int curr_label = result[i].label;
            int next_label = result[i + 1].label;

            // 检查是否缺失2个以上的点
            if (curr_label - prev_label > 2 || next_label - curr_label > 2) {
                Point2f curr_point = result[i].position;
                Point2f miss_prev_point = Point2f(curr_point.x - avgDistance * 0.7,
                                                  curr_point.y);
                Point2f miss_next_point = Point2f(curr_point.x + avgDistance * 0.7,
                                                  curr_point.y);
                for (size_t j = 0; j < 2; ++j) {//可能正反方向
                    Point2f start;
                    Point2f end;
                    int startLabel = curr_label - 1;
                    int endLabel = curr_label + 1;
                    if (j == 0) {
                        start = miss_prev_point;
                        end = miss_next_point;
                    } else {
                        end = miss_prev_point;
                        start = miss_next_point;
                    }

                    //找出最接近的点位
                    LightPoint startPoint = findLamp(start, avgDistance, true,
                                                     startLabel, false, false);
                    LightPoint endPoint = findLamp(end, avgDistance, true,
                                                   endLabel, false, false);
                    //bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
                    if (startPoint.errorStatus != EMPTY_POINT && endPoint.errorStatus != EMPTY_POINT
                        &&
                        isApproximatelyHorizontal(startPoint.position, curr_point,
                                                  endPoint.position)
                        &&
                        (distance.operator()(startPoint.position, curr_point) -
                         distance.operator()(curr_point, endPoint.position)) < avgDistance / 3
                            ) {
                        result.push_back(startPoint);
                        result.push_back(endPoint);
                        break;
                    }
                }
            }
        }

        // 重新按label排序
        std::sort(result.begin(), result.end(), [](const LightPoint &a, const LightPoint &b) {
            return a.label < b.label;
        });
        LOGD(LOG_TAG, "fillMissingPoints = %d", result.size() - totalPoints.size());
        return result;
    } catch (...) {
        LOGD(LOG_TAG, "fillMissingPoints error");
    }
    return totalPoints;
}

/**处理剩余无序点位*/
void decisionRemainingPoints(LampBeadsProcessor &processor) {
    LOGD(LOG_TAG, "decisionRemainingPoints");
    try {
        int size = processor.totalPoints.size();
        for (int i = 1; i < size - 1; i++) {
            LightPoint curLPoint = processor.totalPoints[i];
            //优先从水平线上找
            LightPoint nextLPoint = processor.totalPoints[i + 1];
            LightPoint lastLPoint = processor.totalPoints[i - 1];
            int nextDiff = nextLPoint.label - curLPoint.label;
            if (nextDiff == 2) {
                //再次处理中间点位
                LightPoint centerP = inferredCenter(processor.averageDistance, nextLPoint,
                                                    curLPoint,
                                                    true);
                if (centerP.errorStatus != EMPTY_POINT && centerP.label < getIcNum() &&
                    centerP.label >= 0) {
                    //往后插入一个点
                    processor.totalPoints.push_back(centerP);
                } else {
                    LightPoint lp = LightPoint(curLPoint.label + 1);
                    Point2f midpoint = Point2f((curLPoint.position.x + nextLPoint.position.x) / 2,
                                               (curLPoint.position.y + nextLPoint.position.y) / 2);
                    lp.position = midpoint;
                    lp.label = curLPoint.label + 1;
                    processor.totalPoints.push_back(lp);
                }
            } else if (nextDiff > 2) {
                LOGD(LOG_TAG,
                     "【补点-X】= %d", curLPoint.label + 1);
                LightPoint inferredPoint = findLamp(curLPoint.position,
                                                    processor.averageDistance / 0.65, false,
                                                    curLPoint.label + 1, true);

                if (inferredPoint.errorStatus != EMPTY_POINT) {
                    processor.totalPoints.push_back(inferredPoint);
                }
            }

            int lastDiff = curLPoint.label - lastLPoint.label;
            if (lastDiff > 2) {
                LightPoint inferredPoint = findLamp(curLPoint.position,
                                                    processor.averageDistance / 0.65, false,
                                                    curLPoint.label - 1, true);
                LOGD(LOG_TAG, "【补点-Z】= %d  averageDistance = %d  errorStatus= %d",
                     curLPoint.label - 1,
                     processor.averageDistance, inferredPoint.errorStatus);
                if (inferredPoint.errorStatus != EMPTY_POINT) {
                    processor.totalPoints.push_back(inferredPoint);
                }
            }
        }
//        LOGE(LOG_TAG, "处理剩余无序点位 补充:  %d totalPoints = %d", processor.totalPoints.size() - size,processor.totalPoints.size());
        if (processor.totalPoints.size() > 2) {
            // 按照y值从小到大排序
            sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);
        }
    } catch (...) {
        LOGE(LOG_TAG, "decisionRemainingPoints error");
    }
}

LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                         bool findErrorPoints) {
    try {
        //下一个值没有，推断点可能位置
        LOGD(LOG_TAG,
             "【Right】推断[下一个] = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
             curLPoint.label + 1, curLPoint.label, lastLPoint.label,
             nextLPoint.label);
        Point2f A = lastLPoint.position;
        Point2f B = curLPoint.position;
        Point2f C = nextLPoint.position;

        //AB-X-C,推断X
        bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
        //如果ABC 不再一个线性方向，则从A的上一个点，lastA-A-B-X是否一个线性方向
        if (!abcHorizontal && i > 2) {
            LightPoint lastLastLPoint = totalPoints[i - 2];
            Point2f lastA = lastLastLPoint.position;
            abcHorizontal = isApproximatelyHorizontal(lastA, A, B);
        }
        if (abcHorizontal) {
            LightPoint inferredPoint = inferredAB2Next(lastLPoint, curLPoint, findErrorPoints);
            if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.label >= 0 &&
                inferredPoint.label < getIcNum()) {
                LOGD(LOG_TAG, "【Right】推断成功：%d i = %d", inferredPoint.label,
                     i);
                totalPoints.insert(totalPoints.begin() + i + 1, inferredPoint);
                if (totalPoints[i + 1].label != inferredPoint.label) {
                    LOGE(LOG_TAG, "-----------插入错误");
                }
                return inferredPoint;
            }
        }
    } catch (...) { LOGE(LOG_TAG, "inferredRight error"); }
    return {EMPTY_POINT};
}

LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                        bool findErrorPoints) {
    try {
        LOGD(LOG_TAG,
             "【Left】推断[上一个]点序号 = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
             curLPoint.label - 1, curLPoint.label, lastLPoint.label,
             nextLPoint.label);
        Point2f A = nextLPoint.position;
        Point2f B = curLPoint.position;
        Point2f C = lastLPoint.position;
        bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
        if (!abcHorizontal && i < totalPoints.size() - 2) {
            LightPoint nextNextLPoint = totalPoints[i + 2];
            Point2f nextA = nextNextLPoint.position;
            abcHorizontal = isApproximatelyHorizontal(nextA, A, B);
            LOGD(LOG_TAG, "【Left】3---ABC非水平,推断BC中间点 nextNextLPoint=%d",
                 nextNextLPoint.label);
        }
        if (abcHorizontal) {
            LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint, findErrorPoints);
            LOGD(LOG_TAG, "【Left】2---ABC水平,推断BC中间点 = %d ", inferredPoint.label);
            if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.label >= 0 &&
                inferredPoint.label < getIcNum()) {

                LOGD(LOG_TAG, "【补点流程C】推断序号成功：%d  i = %d", inferredPoint.label, i);
                totalPoints.insert(totalPoints.begin() + i, inferredPoint);
                if (totalPoints[i].label != inferredPoint.label) {
                    LOGE(LOG_TAG, "2-----------插入错误");
                }
                return inferredPoint;
            }
        }
    } catch (...) { LOGE(LOG_TAG, "inferredLeft error"); }
    return {EMPTY_POINT};
}

/**
 * 推测中间点
 * @param A 后一个点
 * @param B 前一个点
 */
LightPoint
inferredCenter(double avgDistance, LightPoint &A, LightPoint &B, bool findErrorPoints) {
    int lastLightIndex = B.label;
    //只补充中间点
    double diffSegmentLenX = (A.position.x - B.position.x) / 2;
    double diffSegmentLenY = (A.position.y - B.position.y) / 2;
    double normPoint = abs(distanceP(A.position, B.position));
    if (normPoint > avgDistance * 3) {
        LOGE(LOG_TAG,
             "【补点-A】点位间隔过大，暂不补点 normPoint=%f , averageDistance=%d , label=%d",
             normPoint, avgDistance, A.label - 1);
        return {EMPTY_POINT};
    }
    int curLightIndex = lastLightIndex + 1;
    LOGD(LOG_TAG,
         "【Center】推断 %d  cur = %d, last = %d diffX = %f diffY = %f  cur(%f x %f)", curLightIndex,
         A.label,
         lastLightIndex, diffSegmentLenX, diffSegmentLenY, A.position.x, A.position.y);
    int x = B.position.x + diffSegmentLenX;
    int y = B.position.y + diffSegmentLenY;

    Point2f center = Point2f(x, y);

    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);

    //找出最接近的点位
    LightPoint inferredPoint = findLamp(center, distanceMin, true,
                                        curLightIndex, findErrorPoints);
    return inferredPoint;
}


LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, bool findErrorPoints) {
    int diff = A.label - B.label;
    double diffSegmentLenX = (A.position.x - B.position.x) / diff;
    double diffSegmentLenY = (A.position.y - B.position.y) / diff;

    int inferredLightIndex, x, y;
    if (diff > 0) {
        inferredLightIndex = B.label - 1;
        x = B.position.x - diffSegmentLenX;
        y = B.position.y - diffSegmentLenY;
    } else {
        inferredLightIndex = B.label + 1;
        x = B.position.x + diffSegmentLenX;
        y = B.position.y + diffSegmentLenY;
    }

    LOGD(LOG_TAG,
         "当前点：%d  A:%d 推断点：%d  diff : %d  diffSegmentLenX : %f  diffSegmentLenY : %f ",
         B.label, A.label, inferredLightIndex, diff, diffSegmentLenX, diffSegmentLenY);

    Point2f center = Point2f(x, y);
    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);

    //找出最接近的点位
    LightPoint inferredPoint = findLamp(center, distanceMin, true, inferredLightIndex,
                                        findErrorPoints);
    return inferredPoint;
}


/**
 * 从小到大排序
 */
bool compareScore(const LightPoint &p1, const LightPoint &p2) {
    return p1.score < p2.score;
}

bool compareIndex(const LightPoint &p1, const LightPoint &p2) {
    return p1.label < p2.label;
}

/**
 * 获取区域颜色集合
 */
vector<LightPoint>
findColorType(const Mat &src, int stepFrame, const vector<LightPoint> &points,
              vector<Mat> &outMats) {
    vector<LightPoint> result;
    if (src.empty())return result;
    try {
        Mat meanColorMat = src.clone();
        vector<vector<Point>> contours;
//        Mat gray;
//        cvtColor(meanColorMat, gray, COLOR_BGR2GRAY);
//
//        // 预处理
        Mat blurred;
        GaussianBlur(src, blurred, Size(5, 5), 1.5);
//
//        Mat binary;
//        threshold(blurred, binary, 200, 255, THRESH_BINARY);
//
//        // 确保binary是CV_8UC1格式
//        if (binary.type() != CV_8UC1) {
//            binary.convertTo(binary, CV_8UC1);
//        }
//        findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // 转换到HSV色彩空间
        for (auto lPoint: points) {
            Scalar scalar;
            LightPoint lightPoint = meanLightColor(blurred, contours, stepFrame, lPoint,
                                                   meanColorMat);
            result.push_back(lightPoint);
        }
        meanColorMat.release(); //todo: linpeng
//        outMats.push_back(meanColorMat);
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "findColorType 时发生异常: %s", e.what());

    }
    return result;
}

/**
 * 获取区域 hsv 色相
 */
LightPoint
meanLightColor(const Mat &image, const vector<vector<Point>> &contours, int stepFrame,
               const LightPoint &lPoint, Mat &meanColorMat,
               double forceRadius) {
    if (image.empty()) {
        LOGE(LOG_TAG, "meanColor: Error: Image not found!");
        return {};
    }
    try {
        Scalar avgPixelIntensity;
        Mat region;
        CUS_COLOR_TYPE colorType = E_W;
        Rect2i safeR = buildRect(lPoint, image, 8);
        region = image(safeR);
        if (region.empty()) {
            LOGE(LOG_TAG, "region is empty!");
            LightPoint newLp = LightPoint(lPoint.position, lPoint.with, lPoint.height);
            newLp.type = E_W;
            return newLp;
        }

        // 获取区域中最亮的点
        double minVal, maxVal;
        Point minLoc, maxLoc;
        // 确保 region 没有 alpha 通道（如果有）
        cvtColor(region, region, COLOR_BGRA2GRAY); // 转换为 BGR 图像
        minMaxLoc(region, &minVal, &maxVal, &minLoc, &maxLoc);  // 获取最亮的点的坐标

        Point2f center = Point2f(safeR.x + maxLoc.x, safeR.y + maxLoc.y);
        // 以最亮点为圆心，半径为 8 的圆形区域，判断该区域是否为亮绿色或亮红色
        Mat regionReCheck = buildRect(center, image, forceRadius);

        avgPixelIntensity = mean(regionReCheck); // 计算圆形区域的平均颜色

        double blue = avgPixelIntensity[0];
        double green = avgPixelIntensity[1];
        double red = avgPixelIntensity[2];
        double bri = 0;

//        LOGD(LOG_TAG, "blue = %f green = %f red = %f", blue, green, red);

        if (red * 1.1 > green) {//red > blue &&  * 1.1
            colorType = E_RED;
        } else if (green > red) {// && green > blue
            colorType = E_GREEN;
        } else if (forceRadius > 6) {
            LOGV(LOG_TAG, "meanColor= 无法识别");
            regionReCheck = buildRect(center, image, forceRadius - 2);
            if (!regionReCheck.empty()) {
                avgPixelIntensity = mean(regionReCheck);
                blue = avgPixelIntensity[0];
                green = avgPixelIntensity[1];
                red = avgPixelIntensity[2];
//                LOGW(LOG_TAG, "blue = %f green = %f red = %f", blue, green, red);
                if (red * 1.1 > green) {//red > blue &&  * 1.1
                    colorType = E_RED;
                } else if (green > red) {// && green > blue
                    colorType = E_GREEN;
                } else {
                    LOGW(LOG_TAG, "meanColor= 无法识别---2");
                    colorType = E_W;
                }
            }
        }
        if (!meanColorMat.empty()) {
            putText(meanColorMat, to_string(stepFrame), Point(50, 50), FONT_HERSHEY_SIMPLEX,
                    0.7, Scalar(255, 0, 50), 2);
            if (!meanColorMat.empty()) {//绘制测试图片
                bool canNotDraw =
                        center.x < 10 || center.x >= meanColorMat.cols - 10 || center.y < 10 ||
                        center.y >= meanColorMat.rows - 10;
                if (!canNotDraw) {
                    if (colorType == E_RED) {//red > blue &&
                        circle(meanColorMat, center, 9, Scalar(0, 0, 255), 2);
                    } else if (colorType == E_GREEN) {// && green > blue
                        circle(meanColorMat, center, 9, Scalar(0, 255, 0), 2);
                    } else {
                        circle(meanColorMat, center, 9, Scalar(255, 0, 0), 2);
                    }
                }
            }
        }
        LightPoint newLp = LightPoint(lPoint.position, lPoint.with, lPoint.height);
        newLp.type = colorType;
        return newLp;
    }
    catch (...) {
        LOGE(LOG_TAG, "========》 异常2");
        LightPoint newLp = LightPoint(lPoint.position, lPoint.with, lPoint.height);
        newLp.type = E_W;
        return newLp;
    }
}

bool isApproximatelyHorizontal(Point2f A, Point2f B, Point2f C) {
    double slopeBA = (double) (B.y - A.y) / (B.x - A.x);
    double slopeCA = (double) (C.y - A.y) / (C.x - A.x);

    if ((slopeBA > 0 && slopeCA < 0) || (slopeBA < 0 && slopeCA > 0)) {
//        LOGW(LOG_TAG, "error slopeBA = %f  slopeCA = %f  threshold = %f", slopeBA, slopeCA,
//             abs(slopeBA - slopeCA));
        return false;
    }

    // 定义一个阈值，用于判断斜率是否接近水平线的斜率
    double threshold = 0.4;
//    LOGD(LOG_TAG, "slopeAB = %f  slopeBC = %f  threshold = %f", slopeAB, slopeBC,
//         abs(slopeAB - slopeBC));
    if (abs(slopeBA - slopeCA) < threshold) {
        return true;
    } else {
        return false;
    }
}

std::string floatToDouble(float value, int precision = 1) {
    double d = static_cast<double>(value); // 将 float 转换为 double
    std::ostringstream out; // 创建一个字符串流
    out << std::fixed << std::setprecision(precision) << d; // 设置格式并写入字符串流
    return out.str(); // 返回格式化后的字符串
}

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LightPoint> &points, int lightTypeSet) {
    try {
        stringstream ss;
        ss << "[";
        for (int i = 0; i < points.size(); i++) {
            ss << "{";
            ss << "\"x\": " << floatToDouble(points[i].position.x) << ", ";
            ss << "\"y\": " << floatToDouble(points[i].position.y) << ", ";
            ss << "\"index\": " << points[i].label;
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

/**
 * 从非序列疑似灯珠集合中查找点位
 */
LightPoint
findLamp(Point2f &center, double minDistance, bool checkDistance, int inferredLightIndex,
         bool findErrorPoints, bool erase) {
    if (inferredLightIndex > getIcNum()) return {EMPTY_POINT};
    int sequenceType = getNonSequenceType(inferredLightIndex, lightType);
    if (sequenceType == -1) {
        LOGD(TAG_INFERRED, "非推断序号 = %d", inferredLightIndex);
        return {EMPTY_POINT};
    }
    LightPoint findLp = findLampInVector(center, minDistance, checkDistance,
                                         sequenceTypeMap[sequenceType], sequenceType, erase);

    if (findErrorPoints && findLp.errorStatus == EMPTY_POINT) {
        LOGV(TAG_INFERRED, "从错点中找");
        findLp = findLampInVector(center, minDistance, checkDistance, errorSerialVector, 4, erase);
    }
    if (findLp.errorStatus != EMPTY_POINT) {
        findLp.label = inferredLightIndex;
    }
    return findLp;
}


/**
 * 从集合中查找点位
 */
LightPoint findLampInVector(Point2f &center, double minDistance, bool checkDistance,
                            vector<LightPoint> &points, int type, bool erase) {
    // 检查输入条件
    if (points.empty() || (checkDistance && minDistance > 150)) {
        LOGW(LOG_TAG, "找不到推断点, 距离过大 ----> type: %d, points size: %zu", type,
             points.size());
        return {EMPTY_POINT};
    }
    try {
        int selectIndex = -1;
        double distanceTemp = minDistance * 1;

        // 遍历所有点，找到距离最近的点
        for (size_t i = 0; i < points.size(); ++i) {
            const LightPoint &itA = points[i];
            double distance = sqrt((itA.position.x - center.x) * (itA.position.x - center.x) +
                                   (itA.position.y - center.y) * (itA.position.y - center.y));

            if (distance < distanceTemp) {
                distanceTemp = distance;
                selectIndex = static_cast<int>(i); // 更新选中的索引
            }
        }

        // 如果没有找到符合条件的点
        if (selectIndex == -1) {
//            LOGV(LOG_TAG, "未找到满足条件的点");
            return {EMPTY_POINT};
        }

        // 选中的点
        LightPoint selectPoint = points[selectIndex];

        // 如果指定删除，则从点集中移除该点
        if (erase) {
            if (selectIndex >= 0 && selectIndex < static_cast<int>(points.size())) {
                points.erase(points.begin() + selectIndex);
            } else {
                LOGE(LOG_TAG, "erase 超出索引范围: selectIndex = %d, points size = %zu",
                     selectIndex, points.size());
            }
        }

        LOGV(LOG_TAG, "findLampInVector points剩余 = %zu, selectIndex = %d", points.size(),
             selectIndex);
        return selectPoint;
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "异常捕获: %s", e.what());
        return {EMPTY_POINT};
    } catch (...) {
        LOGE(LOG_TAG, "未知异常");
        return {EMPTY_POINT};
    }
}


// 计算两个点的欧氏距离
float calculateDistance(const Point2f &p1, const Point2f &p2) {
    return norm(p1 - p2);
}

// 合并重叠的 LightPoint
std::vector<LightPoint>
mergeOverlappingPoints(const std::vector<LightPoint> &points, float radius,
                       float overlapThreshold) {
    std::vector<bool> merged(points.size(), false); // 标记是否已合并
    std::vector<LightPoint> mergedPoints;

    for (size_t i = 0; i < points.size(); ++i) {
        if (merged[i]) continue;

        bool hasMerged = false;
        for (size_t j = i + 1; j < points.size(); ++j) {
            if (!merged[j] && calculateDistance(points[i].position, points[j].position) <=
                              radius * std::sqrt(overlapThreshold)) {
                // 合并两个点的中心位置
                cv::Point2f newCenter = (points[i].position + points[j].position) * 0.5f;
                LightPoint lp = LightPoint();
                lp.position = newCenter;
                lp.type = points[i].type;
                lp.tfRect = points[i].tfRect;
                mergedPoints.emplace_back(lp);
                merged[i] = true;
                merged[j] = true;
                hasMerged = true;
                break;
            }
        }

        // 如果没有找到可以合并的点，将其单独加入
        if (!hasMerged) {
            mergedPoints.push_back(points[i]);
        }
    }

    return mergedPoints;
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

Rect2i buildRect(const LightPoint lp, const Mat &src, int forceRadius) {
    Rect2i safeR;
    try {
        if (src.empty()) {
            LOGE(LOG_TAG, "buildRect src is empty!");
        }
        float x = lp.position.x; // 指定坐标x
        float y = lp.position.y; // 指定坐标y

        Rect roi = Rect(x, y, lp.with, lp.height);
        if (forceRadius != 0.0) {
            roi.width = forceRadius;
            roi.height = forceRadius;
        } else {
            if (roi.width < 7.0) roi.width = 7.0;
            if (roi.height < 7.0) roi.height = 7.0;
        }
        safeR = safeRect2i(roi, src.size());
//        region = src(safeR);
    } catch (...) {
        LOGE(LOG_TAG, "构建点的矩形失败");
    }
    return safeR;
}

Mat
buildRect(const Point2f position, const Mat &src, int forceRadius) {
    Mat region;
    try {
        if (src.empty()) {
            LOGE(LOG_TAG, "buildRect src is empty!");
        }
        float x = position.x; // 指定坐标x
        float y = position.y; // 指定坐标y

        Rect roi = Rect(x, y, forceRadius, forceRadius);
        if (forceRadius != 0.0) {
            roi.width = forceRadius;
            roi.height = forceRadius;
        } else {
            if (roi.width < 7.0) roi.width = 7.0;
            if (roi.height < 7.0) roi.height = 7.0;
        }
        Rect2i safeR = safeRect2i(roi, src.size());
        region = src(safeR);
    } catch (...) {
        LOGE(LOG_TAG, "构建点的矩形失败");
    }
    return region;
}
