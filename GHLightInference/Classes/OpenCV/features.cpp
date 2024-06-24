#include "features.hpp"
#include "sequence.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

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
int number_of_iterations2 = 150;
int lightType = 0;
int motionTypeSet = MOTION_HOMOGRAPHY;
//得分点集合
vector<LightPoint> pPoints;
vector<Point> pPointXys;
unordered_map<int, vector<LightPoint>> pointsStepMap;
//记录有效帧
unordered_map<int, Mat> frameStepMap;
//识别点的区域范围 4个点
int pointsAreaTop = -1, pointsAreaLeft = -1, pointsAreaRight = -1, pointsAreaBottom = -1;

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
        resize(alignMat, srcResize, newSize);
        frameStepMap[frameStep] = alignMat.clone();
    } else {
        pointsStepMap.clear();
        frameStepMap.clear();
        pPoints.clear();
        pointsAreaTop = -1, pointsAreaLeft = -1, pointsAreaRight = -1, pointsAreaBottom = -1;
        resize(originalMat.clone(), srcResize, newSize);
        alignMat = originalMat.clone();
        frameStepMap[frameStep] = alignMat.clone();
    }
    return srcResize;
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
    if (frameStep == STEP_VALID_FRAME_START) {
        Mat src = frameStepMap[frameStep].clone();
        if (!pPointXys.empty())pPointXys.clear();
        if (resultObjects.empty()) {
            findByContours(src, pPointXys, getIcNum(), outMats);
        } else {
            for (int i = 0; i < resultObjects.size(); i++) {
                LightPoint curPoint = resultObjects[i];
                Rect_<int> rect = curPoint.tfRect;
                Point center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
                curPoint.point2f = center;
                curPoint.with = rect.width;
                curPoint.height = rect.height;
                pPointXys.push_back(center);
            }
        }
        for (int i = 0; i < pPointXys.size(); i++) {
            LightPoint curPoint;
            if (!resultObjects.empty()) {
                curPoint = resultObjects[i];
            } else {
                curPoint = LightPoint();
            }
            Point center = pPointXys[i];
            curPoint.point2f = pPointXys[i];
            pPoints.push_back(curPoint);
            if (i == 0 || pointsAreaTop > center.y) {
                pointsAreaTop = center.y;
            }
            if (i == 0 || pointsAreaBottom < center.y) {
                pointsAreaBottom = center.y;
            }
            if (i == 0 || pointsAreaLeft > center.x) {
                pointsAreaLeft = center.x;
            }
            if (i == 0 || pointsAreaRight < center.x) {
                pointsAreaRight = center.x;
            }
        }
        LOGD(LOG_TAG, "pointsAreaTop=%d,pointsAreaBottom=%d,pointsAreaLeft=%d,pointsAreaRight=%d",
             pointsAreaTop, pointsAreaBottom, pointsAreaLeft, pointsAreaRight);
    }
    if (frameStep == STEP_VALID_FRAME_START && !pPoints.empty()) {
        Mat originalFrame1C = frameStepMap[STEP_VALID_FRAME_START].clone();
        putText(originalFrame1C, "tensorFlow",
                Point(40, 40),
                FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 0, 0), 2);
        for (int i = 0; i < pPoints.size(); i++) {
            LightPoint curPoint = pPoints[i];

            Rect roi;
            pPoints[i].buildRect(originalFrame1C, roi);
            rectangle(originalFrame1C, roi, Scalar(0, 255, 255), 4);
            if (i == pPoints.size() - 1) {
                outMats.push_back(originalFrame1C);
            }
        }
    }

    //定位特征点
    vector<LightPoint> findVector = findColorType(frameStepMap[frameStep], frameStep, pPoints,
                                                  outMats);
    pointsStepMap[frameStep] = findVector;
    LOGD(LOG_TAG, "pointsStepMap frameStep=%d getMaxStepCnt=%d", frameStep, getMaxStepCnt());
    if (pointsStepMap.size() == getMaxStepCnt()) {
        //--------------------------------------- 开始识别 ---------------------------------------
        vector<Point2i> trapezoid4Points;
        vector<LightPoint> points = sortLampBeads(frameStepMap[STEP_VALID_FRAME_START], outMats,
                                                  trapezoid4Points);
        pointsStepMap.clear();
        frameStepMap.clear();
        pPoints.clear();
        //回调函数
        return splicedJson(lightPointsToJson(points), point2iToJson(trapezoid4Points));
    }
    return "";
}

/**
 * 对灯带光点排序
 */

vector<LightPoint>
sortLampBeads(Mat &src, vector<Mat> &outMats, vector<Point2i> &trapezoid4Points) {
    int scoreMin = getScoreMin();
    int scoreMax = getScoreMax();
    int maxFrameStep = getMaxStepCnt();

    LampBeadsProcessor processor = LampBeadsProcessor(scoreMin, scoreMax, maxFrameStep);
    if (pPoints.empty())return processor.totalPoints;

    LOGW(LOG_TAG, "sortLampBeads pPoints=%d   scoreMin=%d , scoreMax = %d ,endStep = %d",
         pPoints.size(), scoreMin, scoreMax, maxFrameStep);

    vector<int> sameColorScore = getSameColorVector();
//    deleteEstablishGroupPoints(src);

    /*统计得分*/
    int ret = statisticalScorePoints(src, outMats, processor);

    /*删除离群点+构建梯形*/
    if (lightType == TYPE_H70CX_3D) {
        Mat trapezoidMat = src.clone();
        ret = getMinTrapezoid(trapezoidMat, processor.pointXys, trapezoid4Points);
        outMats.push_back(trapezoidMat);
    }

    if (ret == 0) {
        LOGE(LOG_TAG, "统计得分失败");
        return processor.totalPoints;
    }
    /*处理分值相同的点*/
    decideSameScorePoint(processor, src, outMats);

    sort(processor.normalPoints.begin(), processor.normalPoints.end(), compareIndex);

    //计算点位平均距离
    double averageDistance = calculateAverageDistance(processor);

    /*推测中间夹点*/
    decisionCenterPoints(processor, src);

    if (processor.totalPoints.size() < 4)return processor.totalPoints;

    //对补全的点进行排序
    sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);

    decisionRightLeftPoints(processor);

    //对补全的点进行排序
    sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);

    //处理剩余无序点位
    decisionRemainingPoints(processor);

    // 按照y值从小到大排序
    sort(processor.totalPoints.begin(), processor.totalPoints.end(), compareIndex);
    if (lightType != TYPE_H682X) {
        deleteDiscontinuousPoints(processor);
    } else {
        int size = processor.totalPoints.size();
        for (int i = 0; i < size; i++) {
            if (i >= processor.totalPoints.size() - 1)continue;
            LightPoint curLPoint = processor.totalPoints[i];
            LightPoint nextLPoint = processor.totalPoints[i + 1];
            LOGD(LOG_TAG, "   i= %d=== curLPoint= %d===nextLPoint= %d", i, curLPoint.lightIndex,
                 nextLPoint.lightIndex);
            int curLightIndex;
            if (i == 0 && curLPoint.lightIndex > 1) {
                curLightIndex = curLPoint.lightIndex - 1;
                LOGD(LOG_TAG, " 推断1====  curLightIndex=%d ", curLightIndex);
                //找出最接近的点位
                LightPoint inferredPoint = findLamp(curLPoint.point2f, averageDistance / 0.45 * 2,
                                                    true,
                                                    curLightIndex, processor);

                if (inferredPoint.errorStatus != EMPTY_POINT)
                    processor.totalPoints.push_back(inferredPoint);
            }
            if (nextLPoint.lightIndex - curLPoint.lightIndex > 1) {
                curLightIndex = curLPoint.lightIndex + 1;
                LOGD(LOG_TAG,
                     " 推断====  curLightIndex=%d redSameVector=%d greenSameVector=%d",
                     curLightIndex, processor.redSameVector.size(),
                     processor.greenSameVector.size());
                //找出最接近的点位
                LightPoint inferredPoint = findLamp(curLPoint.point2f, averageDistance / 0.45 * 2,
                                                    true,
                                                    curLightIndex, processor);

                if (inferredPoint.errorStatus != EMPTY_POINT) {
                    processor.totalPoints.push_back(inferredPoint);
                }
            }

        }
    }

    //输出一张好点图
    vector<Point2i> points;
    Mat dstCircle = src.clone();
    for (int i = 0; i < pPoints.size(); i++) {
        Rect roi;
        pPoints[i].buildRect(src, roi);
        if (pPoints[i].errorStatus != NORMAL) {
            rectangle(dstCircle, roi, Scalar(255, 255, 0, 150), 2);
        } else {
            rectangle(dstCircle, roi, Scalar(0, 0, 0, 150), 2);
        }
    }

    for (int i = 0; i < processor.totalPoints.size(); i++) {
        Point2f center = processor.totalPoints[i].point2f;
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        points.push_back(center);
        Rect roi;
        pPoints[i].buildRect(src, roi);
        rectangle(dstCircle, roi, Scalar(0, 255, 255, 150), 2);
        if (lightType == TYPE_H682X) {
            putText(dstCircle, to_string(processor.totalPoints[i].lightIndex), center,
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(0, 0, 0),
                    1);
        } else {
            putText(dstCircle, to_string(processor.totalPoints[i].lightIndex), center,
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(0, 255, 255),
                    1);
        }
    }

    Mat outMat = dstCircle.clone();

    outMats.push_back(outMat);
    outMats.push_back(dstCircle);
    return processor.totalPoints;
}

/**
 * 统计所有得分
 */
int statisticalScorePoints(Mat &src, vector<Mat> &outMats, LampBeadsProcessor &processor) {
    int scoreMin = processor.scoreMin;
    int scoreMax = processor.scoreMax;
    int maxFrameStep = processor.maxFrameStep;
    vector<int> sameColorScore = getSameColorVector();

    Mat out = src.clone();
    //消除
    vector<int> eraseVector = polyPoints(pPointXys, 3, 2.3, out);
    outMats.push_back(out);
    sort(eraseVector.begin(), eraseVector.end(), std::greater<int>());
    for (int index: eraseVector) {
        auto erasePoint = pPoints.begin() + index;
        erasePoint->errorStatus = ERASE_POINT;
    }

    // 1. 统计分值相同的index
    // 2. 标记异常分点
    int normalIndex = 0;
    /*对得分进行排序*/
    sort(pPoints.begin(), pPoints.end(), compareScore);
    for (int i = 0; i < pPoints.size(); i++) {
        //--------------------绘制--------------------
        Point2f center = pPoints[i].point2f;
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
            processor.errorSerialVector.push_back(pPoints[i]);
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
//        pointsSrc.push_back(center);
        if (score == sameColorScore[0]) {
            processor.redSameVector.push_back(pPoints[i]);
            continue;
        }
        if (score == sameColorScore[1]) {
            processor.greenSameVector.push_back(pPoints[i]);
            continue;
        }
        pPoints[i].score = score;
        //计算序列
        syncLightIndex(pPoints[i], score, lightType);

        if (pPoints[i].lightIndex < 0 || pPoints[i].lightIndex > getIcNum()) {
            processor.errorSerialVector.push_back(pPoints[i]);
        } else if ((lightType == TYPE_H70CX_3D || lightType == TYPE_H70CX_2D) &&
                   pPoints[i].lightIndex % 2 != 0) {
            LOGW(LOG_TAG, "error lightIndex: %d", pPoints[i].lightIndex);
            //70dx的序列是奇数点位，不满足的话就是推错点了
            processor.errorSerialVector.push_back(pPoints[i]);
        } else {
            if (processor.sameSerialNumMap[pPoints[i].lightIndex].empty()) {
                processor.normalPoints.push_back(pPoints[i]);
            }
            processor.sameSerialNumMap[pPoints[i].lightIndex].push_back(normalIndex);
            normalIndex++;
        }
    }

    LOGW(LOG_TAG, "greenSameVector = %d   redSameVector = %d   errorSerialVector = %d",
         processor.greenSameVector.size(),
         processor.redSameVector.size(), processor.errorSerialVector.size());
    return 1;
}

/**
 * 计算点位平均距离
 */
double calculateAverageDistance(LampBeadsProcessor &processor) {
    double averageDistance = 20;
    int averageCnt = 1;
    int diff = 1;
    if (lightType != TYPE_H682X) {
        diff = 2;
    }
    for (int i = 0; i < processor.normalPoints.size() - 1; i++) {
        LightPoint curLPoint = processor.normalPoints[i];
        LightPoint nextLPoint = processor.normalPoints[i + 1];
        int xx = nextLPoint.lightIndex - curLPoint.lightIndex;
        if (xx == diff) {
            averageDistance += norm(nextLPoint.point2f - curLPoint.point2f);
            averageCnt += 1;
        }
    }
    averageDistance = averageDistance / averageCnt;
    processor.averageDistance = averageDistance;
    LOGW(LOG_TAG, "平均距离 averageDistance = %f  averageCnt = %d", averageDistance, averageCnt);
    return averageDistance;
}


/**
 * 删除离群点+构建梯形
 */
void deleteEstablishGroupPoints(Mat &src, vector<Mat> &outMats) {

}

/**
 * 推测中间夹点
 */
void
decisionCenterPoints(LampBeadsProcessor &processor, Mat &src) {
    //补充不连续段落,记录 last 临时存储 -1的原因是为了补0
    int lastLightIndex = -999999;
    LightPoint lastPoint = LightPoint();

    for (int i = 0; i < processor.normalPoints.size(); i++) {
        LightPoint normalPoint = processor.normalPoints[i];
        processor.totalPoints.push_back(normalPoint);
        //与上一个点的间隔
        int diff = normalPoint.lightIndex - lastLightIndex - 1;
        if (diff == 0) {
            int size = processor.sameSerialNumMap[normalPoint.lightIndex].size();
            LOGW(LOG_TAG, "1. 当前点与上一次点重复，size=%d", size);
            continue;
        }
        if (lastLightIndex == -999999 || diff != 1) {
            //标记灯序作为下一次遍历节点
            lastLightIndex = normalPoint.lightIndex;
            lastPoint = normalPoint;
            continue;
        }
        if (int size = processor.sameSerialNumMap[normalPoint.lightIndex].size() > 1) {
//            LOGW(LOG_TAG, "当前点重复，size=%d", size);
        }
        LightPoint centerP = inferredCenter(processor, normalPoint, lastPoint);
        if (centerP.errorStatus != EMPTY_POINT && centerP.lightIndex < getIcNum() &&
            centerP.lightIndex >= 0)
            processor.totalPoints.push_back(centerP);
    }
    LOGD(LOG_TAG, "normalPoints = %d totalPoints = %d", processor.normalPoints.size(),
         processor.totalPoints.size());
}


/**
 * 从红绿固定点和错点中推测点位
 */
void decisionRightLeftPoints(LampBeadsProcessor &processor) {
    bool enable4BeginLeft = true;//起点往前补点
    for (auto it = processor.totalPoints.begin();
         it <= processor.totalPoints.end(); ++it) {
        auto beginLP = processor.totalPoints.begin();
        auto endLP = processor.totalPoints.end();
        if (it == beginLP) {
            LightPoint curLPoint = processor.totalPoints[0];
            LightPoint nextLPoint = processor.totalPoints[1];
            int inferredNextDiff = nextLPoint.lightIndex - curLPoint.lightIndex;
            //第一个点
            if (it->lightIndex > 1 && enable4BeginLeft) {
                LOGD(LOG_TAG, "第1个点之前缺失，begin : %d", it->lightIndex);
                LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint, processor);
                if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0 &&
                    inferredPoint.lightIndex < getIcNum()) {
                    processor.totalPoints.insert(processor.totalPoints.begin(),
                                                 inferredPoint);
                    LOGD(LOG_TAG, "补充点位 = %d  ===》重新遍历，直到往前补点失败",
                         inferredPoint.lightIndex);
                    it--;
                    continue;
                } else {
                    enable4BeginLeft = false;
                }
            }
            if (inferredNextDiff > 1 && processor.totalPoints.size() >= 3) {
                LOGD(LOG_TAG, "第1个点之后有缺失，begin : %d", it->lightIndex);
                LightPoint nextNextLPoint = processor.totalPoints[2];
                bool abcHorizontal = isApproximatelyHorizontal(curLPoint.point2f,
                                                               nextLPoint.point2f,
                                                               nextNextLPoint.point2f);
                if (abcHorizontal) {
                    while (inferredNextDiff > 1) {
                        LightPoint inferredPoint = inferredAB2Next(nextNextLPoint, nextLPoint,
                                                                   processor);
                        if (inferredPoint.errorStatus != EMPTY_POINT &&
                            inferredPoint.lightIndex >= 0 &&
                            inferredPoint.lightIndex < getIcNum()) {

                            LOGD(LOG_TAG, "补充点位 = %d", inferredPoint.lightIndex);
                            //从next（i+1）的前一个插入
                            processor.totalPoints.insert(processor.totalPoints.begin() + 1,
                                                         inferredPoint);
                            if (processor.totalPoints[2].lightIndex !=
                                inferredPoint.lightIndex) {
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
            int i = distance(processor.totalPoints.begin(), it);
            LightPoint curLPoint = processor.totalPoints[i];
            LightPoint nextLPoint = processor.totalPoints[i + 1];
            LightPoint lastLPoint = processor.totalPoints[i - 1];
            //计算下一个点是否缺失
            int inferredNextDiff = nextLPoint.lightIndex - curLPoint.lightIndex;
            bool inferred2Right = true;
            //代表nextPoint角标
            int nextRightAdd = 0;
            int lastLeftAdd = 0;
            while (inferredNextDiff > 1) {
                LightPoint inferredPoint;
                if (inferred2Right) {
                    //首先往右边推断
                    inferredPoint = inferredRight(curLPoint, lastLPoint, nextLPoint, i,
                                                  processor);
                    if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0 &&
                        inferredPoint.lightIndex < getIcNum()) {
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
                    if (index > processor.totalPoints.size() - 2 || index < 1) {
                        inferredNextDiff = 0;
                        continue;
                    }
                    curLPoint = processor.totalPoints[index];
                    if (curLPoint.lightIndex >= getIcNum()) {
                        inferredNextDiff = 0;
                        continue;
                    }
                    nextLPoint = processor.totalPoints[index + 1];
                    lastLPoint = processor.totalPoints[index - 1];

                    LOGD(LOG_TAG,
                         "向左推断，当前点序号=%d, index = %d, lastLPoint = %d, nextLPoint = %d",
                         curLPoint.lightIndex, index, lastLPoint.lightIndex, nextLPoint.lightIndex);

                    inferredPoint = inferredLeft(curLPoint, lastLPoint, nextLPoint, index,
                                                 processor);
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
            if (it->lightIndex > getIcNum())break;
        }
    }
}

/**删除不连续错点*/
void deleteDiscontinuousPoints(LampBeadsProcessor &processor) {
    LOGD(LOG_TAG, "删除不连续错点");
    int size = processor.totalPoints.size();
    vector<int> errorPointIndexVector;
    //补充左侧的走向推断
    for (int i = 1; i < processor.totalPoints.size() - 1; i++) {
        LightPoint curLPoint = processor.totalPoints[i];
        LightPoint nextLPoint = processor.totalPoints[i + 1];
        LightPoint lastLPoint = processor.totalPoints[i - 1];

        double averageDistanceThreshold = processor.averageDistance * 2;
        bool lastContinuous = curLPoint.lightIndex == (lastLPoint.lightIndex + 1) && i > 2 &&
                              lastLPoint.lightIndex ==
                              (processor.totalPoints[i - 2].lightIndex + 1);

        double distance4Last = norm(curLPoint.point2f - lastLPoint.point2f) /
                               max((curLPoint.lightIndex * 1.0 - lastLPoint.lightIndex), 1.0);

        bool nextContinuous =
                curLPoint.lightIndex == (nextLPoint.lightIndex - 1) &&
                i < (processor.totalPoints.size() - 2) &&
                nextLPoint.lightIndex == (processor.totalPoints[i + 2].lightIndex - 1);

        double distance4Next = norm(nextLPoint.point2f - curLPoint.point2f) /
                               max((nextLPoint.lightIndex * 1.0 - curLPoint.lightIndex), 1.0);

        if (lastContinuous && abs(distance4Last) > averageDistanceThreshold) {
            errorPointIndexVector.push_back(i);
        } else if (nextContinuous && abs(distance4Next) > averageDistanceThreshold) {
            errorPointIndexVector.push_back(i);
        } else if (abs(distance4Last) > averageDistanceThreshold &&
                   abs(distance4Next) > averageDistanceThreshold) {
            errorPointIndexVector.push_back(i);
        }
    }

    for (int i = errorPointIndexVector.size() - 1; i >= 0; i--) {
        LOGD(LOG_TAG, "erase index=%d, lightIndex=%d", errorPointIndexVector[i],
             processor.totalPoints[errorPointIndexVector[i]].lightIndex);
        processor.totalPoints.erase(processor.totalPoints.begin() + errorPointIndexVector[i]);
    }
    LOGE(LOG_TAG, "通过连续点位置删除点：%d", processor.totalPoints.size() - size);
}

/**处理剩余无序点位*/
void decisionRemainingPoints(LampBeadsProcessor &processor) {
    int size = processor.totalPoints.size();
    for (int i = 1; i < size - 1; i++) {
        LightPoint curLPoint = processor.totalPoints[i];
        //优先从水平线上找
        LightPoint nextLPoint = processor.totalPoints[i + 1];
        LightPoint lastLPoint = processor.totalPoints[i - 1];
        int nextDiff = nextLPoint.lightIndex - curLPoint.lightIndex;
        if (nextDiff == 2) {
            //再次处理中间点位
            LightPoint centerP = inferredCenter(processor, nextLPoint, curLPoint);
            if (centerP.errorStatus != EMPTY_POINT && centerP.lightIndex < getIcNum() &&
                centerP.lightIndex >= 0) {
                //往后插入一个点
                processor.totalPoints.push_back(centerP);
            }
        } else if (nextDiff > 2) {
            LOGD(LOG_TAG,
                 "【补点-X】= %d", curLPoint.lightIndex + 1);
            LightPoint inferredPoint = findLamp(curLPoint.point2f,
                                                processor.averageDistance / 0.7, false,
                                                curLPoint.lightIndex + 1, processor);

            if (inferredPoint.errorStatus != EMPTY_POINT) {
                processor.totalPoints.push_back(inferredPoint);
            }
        }

        int lastDiff = curLPoint.lightIndex - lastLPoint.lightIndex;
        if (lastDiff > 2) {
            LOGD(LOG_TAG, "【补点-Z】= %d", curLPoint.lightIndex - 1);
            LightPoint inferredPoint = findLamp(curLPoint.point2f,
                                                processor.averageDistance / 0.7, false,
                                                curLPoint.lightIndex - 1, processor);
            if (inferredPoint.errorStatus != EMPTY_POINT) {
                processor.totalPoints.push_back(inferredPoint);
            }
        }
    }
    LOGE(LOG_TAG, "处理剩余无序点位 补充:  %d", processor.totalPoints.size() - size);
}

/**
 * 处理同色得分点
 */
void
decideSameScorePoint(LampBeadsProcessor &processor, Mat &src, vector<Mat> &outMats) {
    Mat samePoint = src.clone();
    LOGV(LOG_TAG, "处理同色得分点");
    /*
     * 1. 计算可信度最高的点
     */
    for (auto it = processor.sameSerialNumMap.begin();
         it != processor.sameSerialNumMap.end(); it++) {
        int serialNum = it->first;
        vector<int> sameScoreIndexVector = it->second;
        if (sameScoreIndexVector.size() > 1) {
            int lastGoodIndex = -1;
            double lastGoodScore = 0;
            LOGD(LOG_TAG, "[处理同序列] 序号 = %d, 同分个数 = %d", serialNum,
                 sameScoreIndexVector.size());

            for (int index: sameScoreIndexVector) {
                LightPoint lp = pPoints[index];
                circle(samePoint, lp.point2f, 8, Scalar(0, 255, 255), 2);
                putText(samePoint, to_string(serialNum), lp.point2f,
                        FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 255), 1);
            }
//            for (int index: sameScoreIndexVector) {
//                LightPoint curPoint = pPoints[index];
//                //根据亮度和饱和度计算 可信度
//                float goodScore = curPoint.brightness;
//                LOGV(LOG_TAG,
//                     "取最高分goodScore: %f lastGoodScore: %f ", goodScore,
//                     lastGoodScore);
//                if (lastGoodIndex == -1) {
//                    lastGoodScore = goodScore;
//                    lastGoodIndex = index;
//                    continue;
//                }
//                if (goodScore < lastGoodScore) { //当前点更好，删除上个点得分
//                    LOGD(LOG_TAG, "delete lastGoodIndex: %d  index: %d", lastGoodIndex,
//                         (pPoints[lastGoodIndex].score - scoreMin));
//                    pPoints[lastGoodIndex].score = -1;
//                    lastGoodScore = goodScore;
//                    lastGoodIndex = index;
//                } else {
//                    //当前点不及上个点，删除当前点
//                    LOGE(LOG_TAG, "delete curIndex: %d ", getLightIndex(pPoints[index].score));
//                    pPoints[index].score = -1;
//                }
//            }
//            LOGD(LOG_TAG, "最终good index : %d index = %d", lastGoodIndex,
//                 (pPoints[lastGoodIndex].score - scoreMin));
        }
    }
    outMats.push_back(samePoint);
}

LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, LampBeadsProcessor &processor) {
    //下一个值没有，推断点可能位置
    LOGD(LOG_TAG,
         "【Right】推断[下一个] = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
         curLPoint.lightIndex + 1, curLPoint.lightIndex, lastLPoint.lightIndex,
         nextLPoint.lightIndex);
    Point2i A = lastLPoint.point2f;
    Point2i B = curLPoint.point2f;
    Point2i C = nextLPoint.point2f;

    //AB-X-C,推断X
    bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
    //如果ABC 不再一个线性方向，则从A的上一个点，lastA-A-B-X是否一个线性方向
    if (!abcHorizontal && i > 2) {
        LightPoint lastLastLPoint = processor.totalPoints[i - 2];
        Point2i lastA = lastLastLPoint.point2f;
        abcHorizontal = isApproximatelyHorizontal(lastA, A, B);
    }
    if (abcHorizontal) {
        LightPoint inferredPoint = inferredAB2Next(lastLPoint, curLPoint, processor);
        if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0 &&
            inferredPoint.lightIndex < getIcNum()) {
            LOGD(LOG_TAG, "【Right】推断成功：%d i = %d", inferredPoint.lightIndex,
                 i);
            processor.totalPoints.insert(processor.totalPoints.begin() + i + 1, inferredPoint);
            if (processor.totalPoints[i + 1].lightIndex != inferredPoint.lightIndex) {
                LOGE(LOG_TAG, "-----------插入错误");
            }
            return inferredPoint;
        }
    }
    return LightPoint(EMPTY_POINT);
}

LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, LampBeadsProcessor &processor) {
    LOGD(LOG_TAG,
         "【Left】推断[上一个]点序号 = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
         curLPoint.lightIndex - 1, curLPoint.lightIndex, lastLPoint.lightIndex,
         nextLPoint.lightIndex);
    Point2i A = nextLPoint.point2f;
    Point2i B = curLPoint.point2f;
    Point2i C = lastLPoint.point2f;
    bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
    if (!abcHorizontal && i < processor.totalPoints.size() - 2) {
        LightPoint nextNextLPoint = processor.totalPoints[i + 2];
        Point2i nextA = nextNextLPoint.point2f;
        abcHorizontal = isApproximatelyHorizontal(nextA, A, B);
        LOGD(LOG_TAG, "【Left】3---ABC水平,推断BC中间点 nextNextLPoint=%d",
             nextNextLPoint.lightIndex);
    }
    if (abcHorizontal) {
        LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint, processor);
        LOGD(LOG_TAG, "【Left】2---ABC水平,推断BC中间点 = %d ", inferredPoint.lightIndex);
        if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0 &&
            inferredPoint.lightIndex < getIcNum()) {

            LOGD(LOG_TAG, "【补点流程C】推断序号成功：%d  i = %d", inferredPoint.lightIndex, i);
            //todo:review是否会角标越界
            processor.totalPoints.insert(processor.totalPoints.begin() + i, inferredPoint);
            if (processor.totalPoints[i].lightIndex != inferredPoint.lightIndex) {
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
LightPoint inferredCenter(LampBeadsProcessor &processor, LightPoint &A, LightPoint &B) {
    int lastLightIndex = B.lightIndex;
    //只补充中间点
    double diffSegmentLenX = (A.point2f.x - B.point2f.x) / 2;
    double diffSegmentLenY = (A.point2f.y - B.point2f.y) / 2;
    double normPoint = abs(distanceP(A.point2f, B.point2f));
    if (lightType != TYPE_H682X && normPoint > processor.averageDistance * 3.5) {
        LOGE(LOG_TAG,
             "【补点-A】点位间隔过大，暂不补点 normPoint=%f , averageDistance=%d , lightIndex=%d",
             normPoint,
             processor.averageDistance, A.lightIndex);
        return LightPoint(EMPTY_POINT);
    }

    LOGD(LOG_TAG,
         "【Center】cur = %d, last = %d diffX = %f diffY = %f", A.lightIndex,
         lastLightIndex, diffSegmentLenX, diffSegmentLenY);
    int curLightIndex = lastLightIndex + 1;
    int x = B.point2f.x + diffSegmentLenX;
    int y = B.point2f.y + diffSegmentLenY;

    Point2i center = Point2i(x, y);

    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);

    //找出最接近的点位
    LightPoint inferredPoint = findLamp(center, distanceMin, true,
                                        curLightIndex, processor);
    return inferredPoint;
}


LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, LampBeadsProcessor &processor) {
    int diff = A.lightIndex - B.lightIndex;
    double diffSegmentLenX = (A.point2f.x - B.point2f.x) / diff;
    double diffSegmentLenY = (A.point2f.y - B.point2f.y) / diff;

    int inferredLightIndex, x, y;
    if (diff > 0) {
        inferredLightIndex = B.lightIndex - 1;
        x = B.point2f.x - diffSegmentLenX;
        y = B.point2f.y - diffSegmentLenY;
    } else {
        inferredLightIndex = B.lightIndex + 1;
        x = B.point2f.x + diffSegmentLenX;
        y = B.point2f.y + diffSegmentLenY;
    }

    LOGD(LOG_TAG,
         "当前点：%d 推断点：%d  diff : %d  diffSegmentLenX : %f  diffSegmentLenY : %f ",
         B.lightIndex, inferredLightIndex, diff, diffSegmentLenX, diffSegmentLenY);

    Point2i center = Point2i(x, y);
    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);

    //找出最接近的点位
    LightPoint inferredPoint = findLamp(center, distanceMin, true,
                                        inferredLightIndex, processor);
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
    searchRegion = Rect(pointsAreaLeft, pointsAreaTop, pointsAreaRight - pointsAreaLeft,
                        pointsAreaBottom - pointsAreaTop);
    int area = searchRegion.width * searchRegion.height;
    if (area < 25) {
        // 假设我们只想在目标图像的一个特定区域内搜索
        LOGE(LOG_TAG, "area < 25,use hard rect");
        searchRegion = Rect(120, 80, 400, 480); // x, y, width, height
    }

    rectangle(mask, searchRegion, Scalar::all(255), FILLED);

    // Convert images to grayscale
    Mat alignedImg;
    Mat im1Src, im2Trans;
    // 转换为灰度图
    cvtColor(src, im1Src, CV_BGR2GRAY);

    cvtColor(trans, im2Trans, CV_BGR2GRAY);

    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations2,
                          termination_eps2);
    findTransformECC(im1Src, im2Trans, warp_matrix, motionTypeSet, criteria);//, mask
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
}

/**
 * 从小到大排序
 */
bool compareScore(const LightPoint &p1, const LightPoint &p2) {
    return p1.score < p2.score;
}

bool compareIndex(const LightPoint &p1, const LightPoint &p2) {
    return p1.lightIndex < p2.lightIndex;
}

/**
 * 获取区域颜色集合
 */
vector<LightPoint>
findColorType(Mat &src, int stepFrame, vector<LightPoint> &points, vector<Mat> &outMats) {
    vector<LightPoint> result;
    Mat meanColorMat = src.clone();
    for (int i = 0; i < points.size(); i++) {
        LightPoint lPoint = points[i];
        Scalar scalar;
        LightPoint lightPoint = meanColor(src, stepFrame, lPoint, meanColorMat);
        result.push_back(lightPoint);
    }
    outMats.push_back(meanColorMat);
    return result;
}

/**
 * 获取区域 hsv 色相
 */
LightPoint meanColor(Mat &src, int stepFrame, LightPoint &lPoint, Mat &meanColorMat) {
    if (src.empty()) {
        LOGE(LOG_TAG, "meanColor(stepFrame=%d): Error: Image not found!", stepFrame);
        return LightPoint();

    }
    Point2f point = lPoint.point2f;
    Rect roi;
    Mat region = lPoint.buildRect(src, roi);

    if (region.empty()) {
        LOGE(LOG_TAG, "region is empty!");
        return lPoint.copyPoint(E_W, Scalar());
    }

    Scalar avgPixelIntensity = mean(region);

    double green = avgPixelIntensity[1];
    double red = avgPixelIntensity[2];

    Mat hsv;
    cvtColor(region, hsv, COLOR_BGR2HSV);
    Scalar mean = cv::mean(hsv);

    CUS_COLOR_TYPE colorType = E_W;
    Scalar color = Scalar(0, 255, 255);
    rectangle(meanColorMat, roi, color,
              2);
    if (red > green) {//red > blue &&
        colorType = E_RED;
        putText(meanColorMat,
                "red", point, FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1);

    } else if (green > red) {// && green > blue
        colorType = E_GREEN;
        putText(meanColorMat,
                "green", point, FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1);
    } else {
        LOGV(LOG_TAG, "meanColor= 无法识别");
        putText(meanColorMat,
                "UnKnow", point, FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1);
    }

    return lPoint.copyPoint(colorType, mean);
}

bool isApproximatelyHorizontal(Point2i A, Point2i B, Point2i C) {
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

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LightPoint> &points) {
    stringstream ss;
    ss << "[";
    for (int i = 0; i < points.size(); i++) {
        ss << "{";
        ss << "\"x\": " << points[i].point2f.x << ", ";
        ss << "\"y\": " << points[i].point2f.y << ", ";
        ss << "\"index\": " << points[i].lightIndex << ", ";
        ss << "\"tfScore\": " << points[i].tfScore;
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
    ss << "\"lightPoints\": " << a << ", ";
    ss << "\"trapezoidalPoints\": " << b << "";
    ss << "}";
    return ss.str();
}

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2i> &points) {
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
        ss << "\"x\": " << points[i].x << ", ";
        ss << "\"y\": " << points[i].y << "";
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
LightPoint findLamp(Point2i &center, double minDistance, bool checkDistance, int inferredLightIndex,
                    LampBeadsProcessor &processor) {
    bool isGreen = ((inferredLightIndex + 1) / 2) % 2 == 0;
    if (lightType == TYPE_H682X) {
        int greenRet = checkIsGreen(inferredLightIndex);
        if (greenRet == -1) {
            LOGE(LOG_TAG, "非推断序号");
            return LightPoint(EMPTY_POINT);
        }
        isGreen = greenRet == 1;
    }
    vector<LightPoint> points;
    LightPoint findLp;
    if (isGreen) {
        findLp = findLampInVector(center, minDistance, checkDistance, processor.greenSameVector);
    } else {
        findLp = findLampInVector(center, minDistance, checkDistance, processor.redSameVector);
    }

    if (findLp.errorStatus == EMPTY_POINT) {
        findLp = findLampInVector(center, minDistance, checkDistance, processor.errorSerialVector);
    }
    if (findLp.errorStatus != EMPTY_POINT) {
        findLp.lightIndex = inferredLightIndex;
    }
    return findLp;
}


/**
 * 从集合中查找点位
 */
LightPoint findLampInVector(Point2i &center, double minDistance, bool checkDistance,
                            vector<LightPoint> &points) {
    if (checkDistance && minDistance > 150 && lightType != TYPE_H682X) {
        LOGE(LOG_TAG, "找不到推断点,距离过大");
        return LightPoint(EMPTY_POINT);
    }
    int selectIndex = -1;
    double distanceTemp = minDistance * 0.45;
    for (int i = 0; i < points.size(); i++) {
        LightPoint itA = points[i];
        int contrastX = itA.point2f.x;
        int contrastY = itA.point2f.y;
        double distance = sqrt((contrastX - center.x) * (contrastX - center.x) +
                               (contrastY - center.y) * (contrastY - center.y));
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
//    LOGV(LOG_TAG, "points剩余  = %d", points.size());
    return selectPoint;
}
