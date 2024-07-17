#include "features.hpp"
#include "sequence.hpp"
#include "anomaly.cpp"
#include "select.hpp"
#include "interpolate682x.hpp"
#include "interpolate70cx.hpp"
#include "inferredp.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#include <setjmp.h>
#include <iomanip>
#include <sstream>

jmp_buf jump_buffer;

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

//对齐精度
double termination_eps2 = 1e-4;
int number_of_iterations2 = 80;
int lightType = 0;
int motionTypeSet = MOTION_HOMOGRAPHY;
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

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat) {
    Mat srcResize, alignMat;
    // 指定缩放后的尺寸
    Size newSize(640, 640);
    if (frameStep > STEP_VALID_FRAME_START) {
        Mat originalMatClone = originalMat.clone();
        alignMat = alignImg(frameStepMap[STEP_VALID_FRAME_START], originalMatClone, false);
        if (alignMat.empty()) {
            return alignMat;
        }
        resize(alignMat, srcResize, newSize);
        frameStepMap[frameStep] = alignMat.clone();
    } else {
        release();
        resize(originalMat.clone(), srcResize, newSize);
        alignMat = originalMat.clone();
        frameStepMap[frameStep] = alignMat.clone();
    }
    return srcResize;
}

/**
 * 处理信号异常值
 * @param signal
 */
void signal_handler(int signal) {
    LOGE(LOG_TAG, "====exception====signal_handler");
    longjmp(jump_buffer, 1);
}


/**释放资源*/
void release() {
    pointsStepMap.clear();
    frameStepMap.clear();
    pPointXys.clear();
    pPointXys.shrink_to_fit();
    pPoints.clear();
    pPoints.shrink_to_fit();
    sequenceTypeMap.clear();
    errorSerialVector.clear();
    errorSerialVector.shrink_to_fit();
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
            if (lightTypeP == TYPE_H682X) {
                //面状灯
                findNoodleLamp(src, pPointXys, resultObjects, outMats);
                LOGD(LOG_TAG, "findNoodleLamp resultObjects %d", resultObjects.size());
            } else if (resultObjects.empty()) {
                findByContours(src, pPointXys, resultObjects, getIcNum(), outMats);
            }
            for (int i = 0; i < resultObjects.size(); i++) {
                LightPoint curPoint = resultObjects[i];
                pPointXys.push_back(curPoint.position);
                pPoints.push_back(curPoint);
            }
        }

        if (frameStep == STEP_VALID_FRAME_START && !pPoints.empty()) {
            Mat originalFrame1C = frameStepMap[STEP_VALID_FRAME_START].clone();
            putText(originalFrame1C, "tensorFlow",
                    Point(40, 40),
                    FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(0, 0, 0), 2);
            for (int i = 0; i < pPoints.size(); i++) {
                LightPoint curPoint = pPoints[i];
                if (!curPoint.rotatedRect.size.empty()) {
                    Point2f vertices[4];
                    // 获取旋转矩形的四个顶点
                    curPoint.rotatedRect.points(vertices);
                    // 绘制旋转矩形
                    for (int j = 0; j < 4; ++j) {
                        cv::line(originalFrame1C, vertices[j], vertices[(j + 1) % 4],
                                 cv::Scalar(0, 255, 0), 2);
                    }
                } else {
                    Rect roi;
                    pPoints[i].buildRect(originalFrame1C, roi);
                    if (!roi.empty())
                        rectangle(originalFrame1C, roi, Scalar(255, 0, 50), 4);
                }
                if (i == pPoints.size() - 1) {
                    outMats.push_back(originalFrame1C);
                }
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常1");
        return "{}";
    }

    //定位特征点
    vector<LightPoint> findVector = findColorType(frameStepMap[frameStep], frameStep, pPoints,
                                                  outMats);
    pointsStepMap[frameStep] = findVector;
    LOGD(LOG_TAG, "pointsStepMap frameStep=%d getMaxStepCnt=%d", frameStep, getMaxStepCnt());
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

void drawPointsMatOut(Mat &src, LampBeadsProcessor &processor, vector<Mat> &outMats) {
    try {
        //输出一张好点图
        vector<Point2f> points;
        Mat dstCircle;
        if (lightType == TYPE_H682X) {
            dstCircle = Mat(2000, 1500, src.type());
        } else {
            dstCircle = src.clone();
        }
        for (int i = 0; i < pPoints.size(); i++) {
            LightPoint lpoint = pPoints[i];
            if (!lpoint.rotatedRect.size.empty()) {
            } else {
                Rect roi;
                pPoints[i].buildRect(src, roi);
                if (pPoints[i].errorStatus != NORMAL) {
                    rectangle(dstCircle, roi, Scalar(255, 255, 0, 150), 2);
                } else {
                    rectangle(dstCircle, roi, Scalar(0, 0, 0, 150), 2);
                }
            }
        }
        for (int i = 0; i < processor.totalPoints.size(); i++) {
            LightPoint lpoint = processor.totalPoints[i];
            Point2f center = lpoint.position;
            center.x = static_cast<int>(center.x);
            center.y = static_cast<int>(center.y);
            points.push_back(center);
            if (!lpoint.rotatedRect.size.empty()) {
                Point2f vertices[4];
                // 获取旋转矩形的四个顶点
                lpoint.rotatedRect.points(vertices);
                // 绘制旋转矩形
                for (int j = 0; j < 4; ++j) {
                    if (processor.totalPoints[i].isInterpolate) {
                        cv::line(dstCircle, vertices[j], vertices[(j + 1) % 4],
                                 cv::Scalar(255, 0, 255, 170), 1);
                    } else {
                        cv::line(dstCircle, vertices[j], vertices[(j + 1) % 4],
                                 cv::Scalar(0, 255, 0), 1);
                    }
                }
                circle(dstCircle, processor.totalPoints[i].startPoint, 3, Scalar(255, 255, 0), 3);
                circle(dstCircle, processor.totalPoints[i].endPoint, 3, Scalar(255, 255, 255), 3);
//                LOGD(LOG_TAG, "draw label = %d ", lpoint.label);
            } else {
                Rect roi;
                pPoints[i].buildRect(src, roi);
                rectangle(dstCircle, roi, Scalar(0, 255, 255, 150), 2);
            }
            if (lightType == TYPE_H682X) {
                if (processor.totalPoints[i].isInterpolate) {
                    putText(dstCircle, to_string(processor.totalPoints[i].label), center,
                            FONT_HERSHEY_SIMPLEX,
                            1,
                            Scalar(0, 255, 255),
                            2);
                } else {
                    putText(dstCircle, to_string(processor.totalPoints[i].label), center,
                            FONT_HERSHEY_SIMPLEX,
                            1,
                            Scalar(0, 0, 255),
                            2);
                }
            } else {
                putText(dstCircle, to_string(processor.totalPoints[i].label), center,
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0, 255, 255),
                        1);
            }
        }
        outMats.push_back(dstCircle);
    } catch (...) {
        LOGE(LOG_TAG, "异常状态11");
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

    /*推测中间夹点*/
    processor.totalPoints = decisionCenterPoints(processor.normalPoints, averageDistance);

    if (processor.totalPoints.size() < 4 && lightType != TYPE_H682X)return processor;

    /*处理分值相同的点*/
    processSamePoints(src, outMats, processor.totalPoints, errorSerialVector, averageDistance,
                      processor.sameSerialNumMap, lightType);

    if (lightType == TYPE_H70CX_3D || lightType == TYPE_H70CX_2D) {
        /*推测中间夹点*/
        detectOutlierPoints(processor.totalPoints, errorSerialVector, averageDistance);
    }
    if (processor.totalPoints.size() > 2) {
        decisionRightLeftPoints(processor.totalPoints);

        //对补全的点进行排序
        sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);

        if (lightType != TYPE_H682X) {
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
        }

    }

    //处理剩余无序点位
    decisionRemainingPoints(processor);

    if (lightType != TYPE_H682X) {
        /*删除离群点+构建梯形*/
        if (lightType == TYPE_H70CX_3D) {
            Mat trapezoidMat = src.clone();
            vector<Point2f> point4Trapezoid;
            for (int i = 0; i < processor.totalPoints.size(); i++) {
                point4Trapezoid.push_back(processor.totalPoints[i].position);
            }
            LOGD(LOG_TAG, "point4Trapezoid = %d", point4Trapezoid.size());
            ret = getMinTrapezoid(trapezoidMat, point4Trapezoid, trapezoid4Points);
            if (ret != 1) {
                LOGE(LOG_TAG, "构建梯形异常");
            }
            //补全小的缺失
//            processor.totalPoints = interpolatePoints3D(processor.totalPoints);
            //计算所有中断的组
            vector<Group> groups = groupLightPoints(processor.totalPoints);
            //计算所有中断的组的间隔，缺失端时前段的1.5倍，则视作有问题
            vector<GapInfo> gapInfos = analyzeGaps(groups);

            outMats.push_back(trapezoidMat);
        } else {
//            processor.totalPoints = completeLightPoints2D(processor.totalPoints, getIcNum());
            //    // 使用多项式拟合补全点
            vector<LightPoint> polyPoints = interpolateAndExtrapolatePoints(processor.totalPoints,
                                                                            getIcNum());
            Mat image = src.clone();
            drawPolynomialPoints(image, polyPoints, cv::Scalar(0, 255, 0));  // 绿色
            outMats.push_back(image);
            processor.totalPoints = polyPoints;
        }
    } else {
        try {
            int size = processor.totalPoints.size();
            for (int i = 0; i < size; i++) {
                if (i >= processor.totalPoints.size() - 1)continue;
                LightPoint curLPoint = processor.totalPoints[i];
                LightPoint nextLPoint = processor.totalPoints[i + 1];

                int curLightIndex;
                if (i == 0 && curLPoint.label > 1) {
                    curLightIndex = curLPoint.label - 1;
                    LOGD(LOG_TAG, " 推断1====  curLightIndex=%d ", curLightIndex);
                    //找出最接近的点位
                    LightPoint inferredPoint = findLamp(curLPoint.position,
                                                        averageDistance / 0.45 * 2,
                                                        true,
                                                        curLightIndex, true);

                    if (inferredPoint.errorStatus != EMPTY_POINT)
                        processor.totalPoints.push_back(inferredPoint);
                }
                if (nextLPoint.label - curLPoint.label > 1) {
                    curLightIndex = curLPoint.label + 1;
                    LOGD(LOG_TAG,
                         " 推断====  curLightIndex=%d ", curLightIndex);
                    //找出最接近的点位
                    LightPoint inferredPoint = findLamp(curLPoint.position,
                                                        averageDistance / 0.45 * 2,
                                                        true,
                                                        curLightIndex, true);

                    if (inferredPoint.errorStatus != EMPTY_POINT) {
                        processor.totalPoints.push_back(inferredPoint);
                    }
                }

            }
        } catch (...) {
            LOGE(LOG_TAG, "异常装10");
        }
        int size = processor.totalPoints.size();

        int totalCount = getIcNum(); // 期望的总矩形数
        float targetWidth = 20;
        float targetHeight = 270;

        if (getIcNum() > 10) {
            vector<LightPoint> pointMin;
            vector<LightPoint> pointMax;
            for (auto &item: processor.totalPoints) {
                if (item.label < 10) {
                    pointMin.push_back(item);
                } else {
                    pointMax.push_back(item);
                }
            }

            LOGW(LOG_TAG, "pointMax = %d pointMin = %d", pointMax.size(), pointMin.size());
            pointMin = interpolateAndExtrapolatePoints(src, pointMin, 0, getIcNum() / 2, 2,
                                                       targetWidth,
                                                       targetHeight);
            pointMax = interpolateAndExtrapolatePoints(src, pointMax, getIcNum() / 2,
                                                       getIcNum(), 2, targetWidth,
                                                       targetHeight);
            LOGW(LOG_TAG, "2 pointMax = %d pointMin = %d", pointMax.size(), pointMin.size());
            // 合并两个集合
            vector<LightPoint> mergedVec;
            mergedVec.reserve(pointMin.size() + pointMax.size()); // 预先分配足够的空间
            mergedVec.insert(mergedVec.end(), pointMin.begin(), pointMin.end());
            mergedVec.insert(mergedVec.end(), pointMax.begin(), pointMax.end());
            processor.totalPoints = mergedVec;
        } else {
            processor.totalPoints = interpolateAndExtrapolatePoints(src, processor.totalPoints, 0,
                                                                    totalCount,
                                                                    2, targetWidth,
                                                                    targetHeight);
        }
        LOGD(LOG_TAG, "h682x推断条数 size = %d add = %d ", size,
             processor.totalPoints.size() - size);
    }

    drawPointsMatOut(src, processor, outMats);
    return processor;
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
    if (lightType != TYPE_H682X) {
        Mat out = src.clone();
        vector<int> eraseVector = polyPoints(pPointXys, 3, 2.3, out);
        outMats.push_back(out);
        sort(eraseVector.begin(), eraseVector.end(), std::greater<int>());
        for (int index: eraseVector) {
            auto erasePoint = pPoints.begin() + index;
            erasePoint->errorStatus = ERASE_POINT;
        }
    }
    sequenceTypeMap.clear();
    for (int i = 0; i < 5; i++) {
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
        if (pPoints[i].errorStatus == ERASE_POINT) {
            errorSerialVector.push_back(pPoints[i]);
            continue;
        }
        if (score < scoreMin) {
            LOGW(LOG_TAG, "异常分值(<scoreMin=%d)，endStep = %d，i = %d，index = %d", scoreMin,
                 maxFrameStep, i,
                 (score - scoreMin));
            continue;
        }
        if (score > scoreMax) {
            LOGW(LOG_TAG, "异常分值(>scoreMax=%d)，i=%d，index=%d", scoreMax, i, (score - scoreMin));
            continue;
        }
        if (score == sameColorScore[0]) {
            sequenceTypeMap[0].push_back(pPoints[i]);
            continue;
        }
        if (score == sameColorScore[1]) {
            LOGW(LOG_TAG, "sequenceTypeMap-1 label: %d", pPoints[i].label);
            sequenceTypeMap[1].push_back(pPoints[i]);
            continue;
        }
        if (score == getScoreMax() - 1 && lightType != TYPE_H682X) {
            LOGW(LOG_TAG, "sequenceTypeMap-2 label: %d", pPoints[i].label);
            sequenceTypeMap[2].push_back(pPoints[i]);
            continue;
        }
        if (score == (getScoreMax() - 2) && lightType != TYPE_H682X) {
            LOGW(LOG_TAG, "sequenceTypeMap-3 label: %d", pPoints[i].label);
            sequenceTypeMap[3].push_back(pPoints[i]);
            continue;
        }
        if (score == (getScoreMax() - 3) && lightType != TYPE_H682X) {
            LOGW(LOG_TAG, "sequenceTypeMap-4 label: %d", pPoints[i].label);
            sequenceTypeMap[4].push_back(pPoints[i]);
            continue;
        }
        pPoints[i].score = score;
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
            LOGD(LOG_TAG, "same = %d", indices[0].label);
            samePointsSize++;
        }
    }
    LOGW(LOG_TAG,
         "normalPoints = %d  samePoints = %d  redSameVector = %d  greenSameVector = %d   max = %d   max2 = %d   max3 = %d   errorSerialVector = %d",
         processor.normalPoints.size(),
         samePointsSize,
         sequenceTypeMap[0].size(),
         sequenceTypeMap[1].size(),
         sequenceTypeMap[2].size(),
         sequenceTypeMap[3].size(),
         sequenceTypeMap[4].size(),
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
    int diff = 1;
    if (lightType != TYPE_H682X) {
        diff = 2;
    }
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
    std::sort(distanceList.begin(), distanceList.end());

    // 计算要过滤的数量
    size_t numToRemoveMin = distanceList.size() / 4;
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
            std::cerr << "Error: Cannot maintain more than 5 elements after filtering."
                      << std::endl;
            return 65;
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
    for (size_t i = 0; i < points.size(); ++i) {
        if (expectedLabel < points[i].label) {
            int missingCount = points[i].label - expectedLabel;
            int starIndex = (i == 0) ? i : i - 1;
            LightPoint startLPoint = (i == 0) ? points[i] : points[i - 1];
            LightPoint endLPoint = points[i];
            Point2f startPoint = startLPoint.position;
            Point2f endPoint = endLPoint.position;

            if (missingCount == 1 && i > 0) {
                // 处理中间缺失一个点的情况
                LightPoint interpolated = inferredCenter(averageDistance, endLPoint, startLPoint,
                                                         false);
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

//    int lastLightIndex = -999999;
//    LightPoint lastPoint = LightPoint();
//    vector<LightPoint> output;
//    for (int i = 0; i < input.size(); i++) {
//        LightPoint normalPoint = input[i];
//        output.push_back(normalPoint);
//        //与上一个点的间隔
//        int diff = normalPoint.label - lastLightIndex - 1;
//        if (lastLightIndex == -999999 || diff < 1 || diff > 1) {
//            //标记灯序作为下一次遍历节点
//            lastLightIndex = normalPoint.label;
//            lastPoint = normalPoint;
//            continue;
//        }
//
//        LightPoint centerP = inferredCenter(averageDistance, normalPoint, lastPoint,
//                                            false);
//        if (centerP.errorStatus != EMPTY_POINT && centerP.label < getIcNum() &&
//            centerP.label >= 0)
//            output.push_back(centerP);
//        lastLightIndex = normalPoint.label;
//        lastPoint = normalPoint;
//    }
//    LOGD(LOG_TAG, "normalPoints = %d totalPoints = %d", input.size(),
//         output.size());
//    return output;

//
//    vector<LightPoint> result;
//    int expectedLabel = 0;
//    vector<LightPoint> points = processor.normalPoints;
//    sort(points.begin(), points.end(), [](const LightPoint &a, const LightPoint &b) {
//        return a.label < b.label;
//    });
//    for (size_t i = 0; i < points.size(); ++i) {
//        if (expectedLabel < points[i].label) {
//            int missingCount = points[i].label - expectedLabel;
//            int starIndex = (i == 0) ? i : i - 1;
//            LightPoint startLPoint = (i == 0) ? points[i] : points[i - 1];
//            LightPoint endLPoint = points[i];
//            Point2f startPoint = startLPoint.position;
//            Point2f endPoint = endLPoint.position;
//
//            if (missingCount == 1 && i > 0) {
//                // 处理中间缺失一个点的情况
//                LightPoint interpolated = findMostLikelyCenter(startLPoint, endLPoint,
//                                                               expectedLabel, lightType,
//                                                               sequenceTypeMap, averageDistance);
//                if (interpolated.errorStatus != EMPTY_POINT) {
//                    result.push_back(interpolated);
//                } else {
//                    //当点位的前面或者后面是连续的点时，自动补全中间这个点
//                    try {
//                        if (canBelievedAB(startPoint, endPoint, points, i, averageDistance)) {
//                            LightPoint lp = LightPoint(expectedLabel);
//                            cv::Point2f midpoint;
//                            midpoint.x = (startPoint.x + endPoint.x) / 2;
//                            midpoint.y = (startPoint.y + endPoint.y) / 2;
//                            lp.position = midpoint;
//                            result.push_back(lp);
//                            LOGD(LOG_TAG, "canBelievedAB expectedLabel = %d  midpoint= %f - %f",
//                                 expectedLabel, midpoint.x, midpoint.y);
//                        }
//                    } catch (...) {
//                        LOGE(LOG_TAG, "decisionCenterPoints error!");
//                    }
//                }
//            } else {
//                //大于一个点的情况，从缺失2头往中间推断
//                // 只推断两个点：expectedLabel 和 points[i].label - 1
//                if (starIndex >= 1 && canBelievePrePre(points, i, averageDistance)) {
//                    //推断expectedLabel
//                    LightPoint pre = points[starIndex - 1];
//                    LightPoint interpolatedLp = findExtensionPointAB2C(pre,
//                                                                       startLPoint,
//                                                                       expectedLabel,
//                                                                       lightType,
//                                                                       sequenceTypeMap,
//                                                                       averageDistance);
//                    if (interpolatedLp.errorStatus != EMPTY_POINT && interpolatedLp.label > 0 &&
//                        interpolatedLp.label < getIcNum()) {
//                        LOGD(LOG_TAG, "1---推断expectedLabel %d", expectedLabel);
//                        result.push_back(interpolatedLp);
//                    } else {
//                        LOGW(LOG_TAG, "1---推断expectedLabel error %d", expectedLabel);
//                    }
//                }
//                if (i < points.size() - 1 && canBelieveNextNext(points, i, averageDistance)) {
//                    //推断points[i].label - 1
//                    LightPoint next = points[i + 1];
//                    LightPoint interpolatedLp = findExtensionPointAB2C(next,
//                                                                       endLPoint,
//                                                                       points[i].label - 1,
//                                                                       lightType,
//                                                                       sequenceTypeMap,
//                                                                       averageDistance);
//                    if (interpolatedLp.errorStatus != EMPTY_POINT && interpolatedLp.label > 0 &&
//                        interpolatedLp.label < getIcNum()) {
//                        LOGD(LOG_TAG, "2---推断expectedLabel %d", points[i].label - 1);
//                        result.push_back(interpolatedLp);
//                    } else {
//                        LOGW(LOG_TAG, "2---推断expectedLabel error %d", points[i].label - 1);
//                    }
//                }
//            }
//        }
//        result.push_back(points[i]);
//        expectedLabel = points[i].label + 1;
//    }
//    processor.totalPoints = result;
//    LOGD(LOG_TAG, "推测中间夹点 normalPoints = %d totalPoints = %d", processor.normalPoints.size(),
//         processor.totalPoints.size());
}


/**
 * 从红绿固定点和错点中推测点位
 */
void decisionRightLeftPoints(vector<LightPoint> &totalPoints) {
    LOGD(LOG_TAG, "decisionRightLeftPoints");
    try {
        sort(totalPoints.begin(), totalPoints.end(),
             [](const LightPoint &a, const LightPoint &b) { return a.label < b.label; });

        bool enable4BeginLeft = true;//起点往前补点
        for (auto it = totalPoints.begin();
             it <= totalPoints.end(); ++it) {
            auto beginLP = totalPoints.begin();
            auto endLP = totalPoints.end();
            if (it == beginLP) {
                LightPoint curLPoint = totalPoints[0];
                LightPoint nextLPoint = totalPoints[1];
                int inferredNextDiff = nextLPoint.label - curLPoint.label;
                //第一个点
                if (it->label > 1 && enable4BeginLeft) {
                    LOGD(LOG_TAG, "第1个点之前缺失，begin : %d", it->label);
                    LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint, true);
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
                                                                       true);
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
                                                      totalPoints, true);
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
                                                     totalPoints, true);
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
                it = it + offset;
                if (it->label > getIcNum())break;
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "异常状态13");
    }
}


/**处理剩余无序点位*/
void decisionRemainingPoints(LampBeadsProcessor &processor) {
    LOGD(LOG_TAG, "decisionRemainingPoints");
    int size = processor.totalPoints.size();
    for (int i = 1; i < size - 1; i++) {
        LightPoint curLPoint = processor.totalPoints[i];
        //优先从水平线上找
        LightPoint nextLPoint = processor.totalPoints[i + 1];
        LightPoint lastLPoint = processor.totalPoints[i - 1];
        int nextDiff = nextLPoint.label - curLPoint.label;
        if (nextDiff == 2) {
            //再次处理中间点位
            LightPoint centerP = inferredCenter(processor.averageDistance, nextLPoint, curLPoint,
                                                true);
            if (centerP.errorStatus != EMPTY_POINT && centerP.label < getIcNum() &&
                centerP.label >= 0) {
                //往后插入一个点
                processor.totalPoints.push_back(centerP);
            }
        } else if (nextDiff > 2) {
            LOGD(LOG_TAG,
                 "【补点-X】= %d", curLPoint.label + 1);
            LightPoint inferredPoint = findLamp(curLPoint.position,
                                                processor.averageDistance / 0.45, false,
                                                curLPoint.label + 1, true);

            if (inferredPoint.errorStatus != EMPTY_POINT) {
                processor.totalPoints.push_back(inferredPoint);
            }
        }

        int lastDiff = curLPoint.label - lastLPoint.label;
        if (lastDiff > 2) {
            LOGD(LOG_TAG, "【补点-Z】= %d  averageDistance = %d", curLPoint.label - 1,
                 processor.averageDistance);
            LightPoint inferredPoint = findLamp(curLPoint.position,
                                                processor.averageDistance / 0.45, false,
                                                curLPoint.label - 1, true);
            if (inferredPoint.errorStatus != EMPTY_POINT) {
                processor.totalPoints.push_back(inferredPoint);
            }
        }
    }
    LOGE(LOG_TAG, "处理剩余无序点位 补充:  %d", processor.totalPoints.size() - size);
    if (processor.totalPoints.size() > 2) {
        // 按照y值从小到大排序
        sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);
    }
}

LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                         bool findErrorPoints) {
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
    return LightPoint(EMPTY_POINT);
}

LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                        bool findErrorPoints) {
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
        LOGD(LOG_TAG, "【Left】3---ABC水平,推断BC中间点 nextNextLPoint=%d",
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
    return LightPoint(EMPTY_POINT);
}

/**
 * 推测中间点
 * @param A 后一个点
 * @param B 前一个点
 */
LightPoint
inferredCenter(int avgDistance, LightPoint &A, LightPoint &B, bool findErrorPoints) {
    int lastLightIndex = B.label;
    //只补充中间点
    double diffSegmentLenX = (A.position.x - B.position.x) / 2;
    double diffSegmentLenY = (A.position.y - B.position.y) / 2;
    double normPoint = abs(distanceP(A.position, B.position));
    if (lightType != TYPE_H682X && normPoint > avgDistance * 3.5) {
        LOGE(LOG_TAG,
             "【补点-A】点位间隔过大，暂不补点 normPoint=%f , averageDistance=%d , label=%d",
             normPoint, avgDistance, A.label - 1);
        return LightPoint(EMPTY_POINT);
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
 * 对齐图片
 */
Mat alignImg(Mat &src, Mat &trans, bool back4Matrix) {
    if (src.empty()) {
        LOGE(LOG_TAG, "===========src empty===========");
        return trans;
    }
    if (trans.empty()) {
        LOGE(LOG_TAG, "===========trans empty===========");
        return src;
    }
    Mat alignedImg;
    try {
        Mat warp_matrix;
        if (motionTypeSet == MOTION_AFFINE) {
            warp_matrix = Mat::eye(2, 3, CV_32F);
        } else if (motionTypeSet == MOTION_HOMOGRAPHY) {//MOTION_HOMOGRAPHY 耗时更久
            warp_matrix = Mat::eye(3, 3, CV_32F);
        } else {
            //MOTION_EUCLIDEAN
        }
        // 降低图像分辨率
        // 创建掩膜，指定搜索区域
        Mat mask = Mat::zeros(trans.size(), CV_8UC1);
        Rect searchRegion;
//        searchRegion = Rect(pointsAreaLeft, pointsAreaTop, pointsAreaRight - pointsAreaLeft,
//                            pointsAreaBottom - pointsAreaTop);
        int area = searchRegion.width * searchRegion.height;
        if (area < 25 && !searchRegion.empty()) {
            // 假设我们只想在目标图像的一个特定区域内搜索
            LOGE(LOG_TAG, "area < 25,use hard rect");
            searchRegion = Rect(120, 80, 400, 480); // x, y, width, height
        }

        rectangle(mask, searchRegion, Scalar::all(255), FILLED);

        Mat im1Src, im2Trans;
        // 转换为灰度图
        cvtColor(src, im1Src, CV_BGR2GRAY);

        cvtColor(trans, im2Trans, CV_BGR2GRAY);

        TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations2,
                              termination_eps2);
        double ecc = findTransformECC(im1Src, im2Trans, warp_matrix, motionTypeSet,
                                      criteria);//, mask
        double alignmentQuality = 1.0 / (1.0 + ecc);
        LOGW(LOG_TAG, "ecc = %f  alignmentQuality = %f", ecc, alignmentQuality);
        if (motionTypeSet == MOTION_HOMOGRAPHY) {
            warpPerspective(trans, alignedImg, warp_matrix, trans.size(),
                            INTER_LINEAR + WARP_INVERSE_MAP);
        } else {
            warpAffine(trans, alignedImg, warp_matrix, trans
                    .size(), INTER_LINEAR + WARP_INVERSE_MAP);
        }
        if (back4Matrix) {
            return warp_matrix;
        }
//    // 在图像上绘制矩形
//    rectangle(alignedImg, searchRegion, Scalar(255, 255, 0), 2);
        return alignedImg;
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常4");
        return alignedImg;
    }
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
findColorType(Mat &src, int stepFrame, vector<LightPoint> &points, vector<Mat> &outMats) {
    vector<LightPoint> result;
    if (src.empty())return result;
    Mat meanColorMat = src.clone();
    // 转换到HSV色彩空间
    cv::Mat image = src.clone();
    for (int i = 0; i < points.size(); i++) {
        LightPoint lPoint = points[i];
        Scalar scalar;
        LightPoint lightPoint = meanColor(image, stepFrame, lPoint, meanColorMat);
        result.push_back(lightPoint);
    }
//    outMats.push_back(meanColorMat);
    return result;
}

/**
 * 获取区域 hsv 色相
 */
LightPoint meanColor(Mat &image, int stepFrame, LightPoint &lPoint, Mat &meanColorMat) {
    if (image.empty()) {
        LOGE(LOG_TAG, "meanColor(stepFrame=%d): Error: Image not found!", stepFrame);
        return LightPoint();
    }
    try {
        Rect roi;
        Scalar avgPixelIntensity;
        Mat region;
        Point2f point = lPoint.position;
        CUS_COLOR_TYPE colorType = E_W;
        if (lightType == TYPE_H682X && !lPoint.rotatedRect.size.empty()) {
            // 创建掩码
            cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
            // 绘制 RotatedRect 到掩码上
            Point2f vertices[4];
            lPoint.rotatedRect.points(vertices);
            vector<cv::Point> contour(vertices, vertices + 4);
            cv::fillPoly(mask, vector<vector<cv::Point>>{contour}, cv::Scalar(255));
            // 提取HSV值
            image.copyTo(region, mask);
            avgPixelIntensity = mean(region, mask);
            roi = lPoint.rotatedRect.boundingRect();
        } else {
            region = lPoint.buildRect(image, roi);
            if (region.empty()) {
                LOGE(LOG_TAG, "region is empty!");
                return lPoint.copyPoint(E_W, Scalar());
            }
            avgPixelIntensity = mean(region);
        }
        double green = avgPixelIntensity[1];
        double red = avgPixelIntensity[2];

        if (red > green) {//red > blue &&
            colorType = E_RED;
        } else if (green > red) {// && green > blue
            colorType = E_GREEN;
        } else {
            LOGV(LOG_TAG, "meanColor= 无法识别");
        }

        if (!meanColorMat.empty()) {//绘制测试图片
            Scalar color = Scalar(0, 255, 255);
            if (!roi.empty())
                rectangle(meanColorMat, roi, color, 2);
            if (red > green) {//red > blue &&
                putText(meanColorMat,
                        "red", point, FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1);

            } else if (green > red) {// && green > blue
                putText(meanColorMat,
                        "green", point, FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1);
            } else {
                putText(meanColorMat,
                        "UnKnow", point, FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1);
            }
        }
        return lPoint.copyPoint(colorType, avgPixelIntensity);
    }
    catch (...) {
        LOGE(LOG_TAG, "========》 异常2");
        return lPoint.copyPoint(E_W, Scalar());
    }
}

bool isApproximatelyHorizontal(Point2f A, Point2f B, Point2f C) {
    double slopeBA = (double) (B.y - A.y) / (B.x - A.x);
    double slopeCA = (double) (C.y - A.y) / (C.x - A.x);

    if ((slopeBA > 0 && slopeCA < 0) || (slopeBA < 0 && slopeCA > 0)) {
        LOGW(LOG_TAG, "error slopeBA = %f  slopeCA = %f  threshold = %f", slopeBA, slopeCA,
             abs(slopeBA - slopeCA));
        return false;
    }

    // 定义一个阈值，用于判断斜率是否接近水平线的斜率
    double threshold = 0.45;
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
    LOGD(LOG_TAG, "lightType = %d", lightType);
    stringstream ss;
    ss << "[";
    for (int i = 0; i < points.size(); i++) {
        ss << "{";
        ss << "\"x\": " << floatToDouble(points[i].position.x) << ", ";
        ss << "\"y\": " << floatToDouble(points[i].position.y) << ", ";
        if (lightTypeSet == TYPE_H682X) {
            ss << "\"startX\": " << floatToDouble(points[i].startPoint.x) << ", ";
            ss << "\"startY\": " << floatToDouble(points[i].startPoint.y) << ", ";
            ss << "\"endX\": " << floatToDouble(points[i].endPoint.x) << ", ";
            ss << "\"endY\": " << floatToDouble(points[i].endPoint.y) << ", ";
        }
//        ss << "\"tfScore\": " << points[i].tfScore << ", ";
        ss << "\"index\": " << points[i].label;
        ss << "}";
        if (i < points.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

string splicedJson(string a, string b) {
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
}

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2f> &points) {
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
}

/**
 * 从非序列疑似灯珠集合中查找点位
 */
LightPoint
findLamp(Point2f &center, double minDistance, bool checkDistance, int inferredLightIndex,
         bool findErrorPoints) {
    if (inferredLightIndex > getIcNum()) return LightPoint(EMPTY_POINT);
    int sequenceType = getNonSequenceType(inferredLightIndex, lightType);
//    LOGE(TAG_INFERRED, "findLamp  sequenceType=%d  inferredLightIndex=%d", sequenceType,
//         inferredLightIndex);
    if (sequenceType == -1) {
        LOGE(TAG_INFERRED, "非推断序号");
        return LightPoint(EMPTY_POINT);
    }
    LightPoint findLp = findLampInVector(center, minDistance, checkDistance,
                                         sequenceTypeMap[sequenceType], sequenceType);

    if (findErrorPoints && findLp.errorStatus == EMPTY_POINT) {
        LOGE(TAG_INFERRED, "从错点中找");
        findLp = findLampInVector(center, minDistance, checkDistance, errorSerialVector, 4);
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
                            vector<LightPoint> &points, int type) {

    if (points.empty() || (checkDistance && minDistance > 150 && lightType != TYPE_H682X)) {
        LOGE(LOG_TAG, "找不到推断点,距离过大----> %d points=%d", type, points.size());
        return LightPoint(EMPTY_POINT);
    }
    try {
        int selectIndex = -1;
        double distanceTemp = minDistance * 0.6;
        for (int i = 0; i < points.size(); i++) {
            LightPoint itA = points[i];
            float contrastX = itA.position.x;
            float contrastY = itA.position.y;
            double distance = sqrt((contrastX - center.x) * (contrastX - center.x) +
                                   (contrastY - center.y) * (contrastY - center.y));
//            LOGD(LOG_TAG, "distance = %f  distanceTemp = %f  contrast=%f x %f", distance,
//                 distanceTemp, contrastX, contrastY);
            if (distance < distanceTemp) {
                distanceTemp = distance;
                selectIndex = i;
            }
        }
        if (selectIndex == -1) {
            return LightPoint(EMPTY_POINT);
        }
        LightPoint selectPoint = points[selectIndex];
        points.erase(points.begin() + selectIndex);
        LOGV(LOG_TAG, "points剩余  = %d  selectIndex= %d", points.size(), selectIndex);
        return selectPoint;
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常3");
        return LightPoint(EMPTY_POINT);
    }
}
