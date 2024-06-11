#include "features.hpp"
#include "sequence.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

/**
 * 得分计算序列
 */
vector <vector<int>> scoreVV = {{0, 1,   2},
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
// Specify the number of iterations.
int number_of_iterations2 = 150;
float fontScale = 0.6;
int radiusCircle2 = 2;
int motionTypeSet = MOTION_HOMOGRAPHY;
//得分点集合
vector <LightPoint> pPoints;
unordered_map<int, vector<LightPoint>> pointsStepMap;
//记录有效帧
unordered_map<int, Mat> frameStepMap;
//识别点的区域范围 4个点
int pointsAreaTop = -1, pointsAreaLeft = -1, pointsAreaRight = -1, pointsAreaBottom = -1;
//记录有效帧
//缺省补全逻辑
COMPLETION_LEVEL completionLevel = COMPLETION_INTERVAL;

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @param outMats 输出流程中的测试图像
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat, vector <Mat> &outMats) {
    Mat srcResize, alignMat;
    // 指定缩放后的尺寸
    cv::Size newSize(640, 640);
    if (frameStep > STEP_VALID_FRAME_START) {
        LOGV(LOG_TAG, "alignMat对齐帧 %d", frameStep);
        Mat originalMatClone = originalMat.clone();
        alignMat = alignImg(frameStepMap[STEP_VALID_FRAME_START], originalMatClone, false);
        resize(alignMat, srcResize, newSize);
        frameStepMap[frameStep] = alignMat.clone();
    } else {
        pointsStepMap.clear();
        frameStepMap.clear();
        pPoints.clear();
        pointsAreaTop = -1, pointsAreaLeft = -1, pointsAreaRight = -1, pointsAreaBottom = -1;

        LOGV(LOG_TAG, "------起始帧------");
//        outMats.push_back(originalMat.clone());
//
//        // 将图像从BGR转换到HSV颜色空间
//        cv::Mat hsv;
//        cv::cvtColor(originalMat, hsv, cv::COLOR_BGR2HSV);
//
//        // 分离HSV通道
//        std::vector<cv::Mat> hsvChannels;
//        cv::split(hsv, hsvChannels);
//        // 定义增强系数
//        double sFactor = 1.2; // 饱和度增强系数
//        double vFactor = 1.3; // 亮度增强系数
//
//        // 创建S和V通道的LUT
//        int lutSize = 256;
//        cv::Mat sLUT(1, lutSize, CV_8UC1);
//        cv::Mat vLUT(1, lutSize, CV_8UC1);
//        for (int i = 0; i < lutSize; ++i) {
//            // 增强饱和度
//            sLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * sFactor);
//            // 增强亮度
//            vLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * vFactor);
//        }
//
//        // 应用LUT到S通道
//        cv::LUT(hsvChannels[1], sLUT, hsvChannels[1]);
//        // 应用LUT到V通道
//        cv::LUT(hsvChannels[2], vLUT, hsvChannels[2]);
//
//        // 合并增强后的通道回HSV图像
//        cv::merge(hsvChannels, hsv);
//
//        // 将HSV图像转换回BGR颜色空间以便显示
//        cv::Mat enhanced;
//        cv::cvtColor(hsv, enhanced, cv::COLOR_HSV2BGR);
//        outMats.push_back(enhanced.clone());
//
//        Mat imageRGB[3];
//        split(enhanced, imageRGB);
//        LOGW(LOG_TAG, "originalMat type:%d", originalMat.type());
//        for (int i = 0; i < 3; i++) {
//            equalizeHist(imageRGB[i], imageRGB[i]);
//        }
//        merge(imageRGB, 3, enhanced);
//
//        Mat sharpened;
//        GaussianBlur(enhanced, sharpened, Size(0, 0), 3);
//        addWeighted(enhanced, 1.5, sharpened, -0.5, 0, sharpened);
//
//        // 创建一个结构元素
//        Mat kernelDilate = getStructuringElement(MORPH_RECT, Size(3, 3));
//        Mat kernelErode = getStructuringElement(MORPH_RECT, Size(1, 1));
//
//        // 对图像进行膨胀操作
//        dilate(enhanced, enhanced, kernelDilate);
//        erode(enhanced, enhanced, kernelErode);
//
//        GaussianBlur(sharpened, enhanced, Size(3, 3), 0);
//
//        outMats.push_back(enhanced.clone());

//        Mat imageRGB[3];
//        Mat equalizedImage;
//        split(originalMat, imageRGB);
//        LOGW(LOG_TAG, "originalMat type:%d", originalMat.type());
//        for (int i = 0; i < 3; i++) {
//            Mat image = imageRGB[i];
//            Ptr<CLAHE> clahe = createCLAHE(3, Size(9, 9));
//            clahe->apply(image, imageRGB[i]);
//        }
//        merge(imageRGB, 3, equalizedImage);
//        Mat sharpened;
//        GaussianBlur(equalizedImage, sharpened, Size(0, 0), 3);
//        addWeighted(equalizedImage, 1.5, sharpened, -0.5, 0, sharpened);
//        GaussianBlur(sharpened, equalizedImage, Size(11, 11), 0);
//
//        outMats.push_back(equalizedImage.clone());
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
String sortStripByStep(int frameStep, vector <LightPoint> &resultObjects, int radiusCircle,
                       vector <Mat> &outMats) {
    if (frameStep == STEP_VALID_FRAME_START && !resultObjects.empty()) {
        LOGD(LOG_TAG, "-------tensorFlow");
        Mat originalFrame1C = frameStepMap[STEP_VALID_FRAME_START].clone();
        putText(originalFrame1C, "tensorFlow",
                Point(40, 40),
                FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 0, 0), 2);
        for (int i = 0; i < resultObjects.size(); i++) {
            LightPoint curPoint = resultObjects[i];
            Rect_<int> rect = curPoint.tfRect;
            Point center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
            if (rect.width < 1 || rect.height < 1) {
                LOGE(LOG_TAG, "-------tensorFlow  error");
            }
            curPoint.point2f = center;
            curPoint.contourArea = rect.width * rect.height;
            circle(originalFrame1C, center, radiusCircle, Scalar(255, 255, 255), 2);
            if (i == resultObjects.size() - 1) {
                outMats.push_back(originalFrame1C);
            }
        }
    }
    if (frameStep == STEP_VALID_FRAME_START) {
        if (resultObjects.empty()) {
            Mat src = frameStepMap[STEP_VALID_FRAME_START].clone();
            vector <Point> points, trapezoidPoints;
            findByContours(src, points, trapezoidPoints, outMats);
            LOGD(LOG_TAG, "-------findByContours points=%d", points.size());
            for (int i = 0; i < points.size(); i++) {
                LightPoint curPoint = LightPoint();
                Point center = points[i];
                curPoint.point2f = points[i];
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
        } else {
            for (int i = 0; i < resultObjects.size(); i++) {
                LightPoint curPoint = resultObjects[i];
                Rect_<int> rect = curPoint.tfRect;
                Point center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
                curPoint.point2f = center;
                curPoint.contourArea = rect.width * rect.height;
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
        }
        LOGD(LOG_TAG, "pointsAreaTop=%d,pointsAreaBottom=%d,pointsAreaLeft=%d,pointsAreaRight=%d",
             pointsAreaTop, pointsAreaBottom, pointsAreaLeft, pointsAreaRight);
    }
    //定位特征点
    pointsStepMap[frameStep] = findColorType(frameStepMap[frameStep], frameStep, pPoints,
                                             outMats);

    if (frameStep == getMaxStepCnt() - 1) {
        //--------------------------------------- 开始识别 ---------------------------------------
        vector <Point2i> trapezoid4Points;
        vector <LightPoint> points = checkLightStrip(frameStepMap[STEP_VALID_FRAME_START], outMats,
                                                     trapezoid4Points);
        pointsStepMap.clear();
        frameStepMap.clear();
        pPoints.clear();
        if (points.empty()) {
            return "";
        }
        //回调函数
        return splicedJson(lightPointsToJson(points), point2iToJson(trapezoid4Points));
    }
    return "";
}

/**
 * 对灯带光点排序
 */
vector <LightPoint>
checkLightStrip(Mat & src, vector < Mat > &outMats, vector < Point2i > &trapezoid4Points) {
    //构建新的集合来存储
    vector <LightPoint> pointsNew;
    int scoreMin = getScoreMin();
    int scoreMax = getScoreMax();
    int maxFrameStep = getMaxStepCnt();
    LOGW(LOG_TAG, "checkLightStrip pPoints=%d   scoreMin=%d , scoreMax = %d ,endStep = %d",
         pPoints.size(), scoreMin, scoreMax, maxFrameStep);
    vector<int> sameColorScore = getSameColorVector();

    // 1. 统计分值相同的index
    // 2. 标记异常分点
    map<int, vector<int>> sameScoreMap;
    vector <LightPoint> redSameVector;
    vector <LightPoint> greenSameVector;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < pPoints.size(); i++) {
        //--------------------绘制--------------------
        Point2f center = pPoints[i].point2f;
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);

        int score = 0;
        for (int step = STEP_VALID_FRAME_START; step < maxFrameStep; step++) {
            score += scoreVV[step][pointsStepMap[step][i].type];
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
//        if (pointsStepMap[1][i].brightness < 4) {
//            LOGW(LOG_TAG, "亮度值过低剔除，index = %d，brightness = %f", getLightIndex(score),
//                 pointsStepMap[1][i].brightness);
//            continue;
//        }
        if (score == sameColorScore[0]) {
            redSameVector.push_back(pPoints[i]);
            continue;
        }
        if (score == sameColorScore[1]) {
            greenSameVector.push_back(pPoints[i]);
            continue;
        }

        pPoints[i].score = score;
        sameScoreMap[score].push_back(i);
    }


    /*
     * 1. 计算可信度最高的点
     */
//    for (auto it = sameScoreMap.begin(); it != sameScoreMap.end(); it++) {
//        int sameScore = it->first;
//        vector<int> sameScoreIndexVector = it->second;
//        if (sameScoreIndexVector.size() > 1) {
//            int lastGoodIndex = -1;
//            double lastGoodScore = 0;
//            LOGV(LOG_TAG, "得分相同 序号 = %d, 同分个数 = %d", getLightIndex(sameScore),
//                 sameScoreIndexVector.size());
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
//        }
//    }

    //--------------------------输出一张好点图--------------------------
    if (pPoints.empty())return pointsNew;
    // 按照y值从小到大排序
    sort(pPoints.begin(), pPoints.end(), compareScore);

    //补充不连续段落,记录 last 临时存储 -1的原因是为了补0
    int lastLightIndex = -999999;
    LightPoint lastPoint = LightPoint();

    for (int i = 0; i < pPoints.size(); i++) {
        LightPoint point = pPoints[i];
        int score = point.score;
        if (score < scoreMin || score > scoreMax) {
            continue;
        }
        if (point.errorStatus != NORMAL) {
            LOGW(LOG_TAG, "丢弃异常状态 i : %d  序号=%d error=%d", i, (score - scoreMin),
                 point.errorStatus);
            continue;
        }
        if (lastLightIndex == -999999) {
            setLightIndex(point, score);
            if (point.lightIndex >= 0) {
                lastLightIndex = point.lightIndex;
                lastPoint = point;
                LOGW(LOG_TAG, "标记点位 = %d   lastLightIndex = %d", point.lightIndex,
                     lastLightIndex);
                if (point.lightIndex < getIcNum() && point.lightIndex % 2 == 0)
                    pointsNew.push_back(point);
            }
            continue;
        }
        setLightIndex(point, point.score);
        int diff = point.lightIndex - lastLightIndex - 1;
        //补充tensorFlow已经识别的点
        if (diff == 1) {
            //推断点可能位置
            double diffSegmentLenX = (point.point2f.x - lastPoint.point2f.x) / (diff + 1);
            double diffSegmentLenY = (point.point2f.y - lastPoint.point2f.y) / (diff + 1);
            LOGW(LOG_TAG,
                 "【补点流程A】需要补点，差值 = %d  当前序号 = %d, 上一个序号 =%d diffSegmentLenX=%f diffSegmentLenY=%f",
                 diff,
                 point.lightIndex, lastLightIndex, diffSegmentLenX, diffSegmentLenY);
            for (int j = 0; j < diff; j++) {
                int curLightIndex = lastLightIndex + j + 1;
                int x = lastPoint.point2f.x + diffSegmentLenX * (j + 1);
                int y = lastPoint.point2f.y + diffSegmentLenY * (j + 1);

                bool isGreen = ((curLightIndex + 1) / 2) % 2 == 0;
                LightPoint inferredPoint;
                Point2i center = Point2i(x, y);

                double distanceMin = sqrt(
                        diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);
                if (distanceMin > 150) {
                    LOGE(LOG_TAG, "1--找不到推断点,距离过大");
                    continue;
                }
                if (!isGreen) {
                    inferredPoint = syncRectPoints(center, distanceMin, redSameVector);
                } else {
                    inferredPoint = syncRectPoints(center, distanceMin, greenSameVector);
                }
                if (inferredPoint.errorStatus != EMPTY_POINT) {
                    inferredPoint.lightIndex = curLightIndex;
                    LOGD(LOG_TAG, "【补点流程A】推断点序号：%d  xy: %d-%d", inferredPoint.lightIndex,
                         inferredPoint.point2f.x, inferredPoint.point2f.y);
                    if (inferredPoint.lightIndex >= 0) {
                        if (point.lightIndex < getIcNum() && point.lightIndex % 2 == 0)
                            pointsNew.push_back(inferredPoint);
                        continue;
                    } else {
                        LOGE(LOG_TAG, "【补点流程A】inferredPoint index 小于0   index=%d  score=%d",
                             inferredPoint.lightIndex, score);
                    }
                }
                LightPoint newPoint = LightPoint(Point2f(x, y), lastPoint.contourArea);
                newPoint.lightIndex = curLightIndex;
                LOGW(LOG_TAG, "【补点流程A】无定位 插入数据 序号 = %d", newPoint.lightIndex);
                if (newPoint.lightIndex >= 0) {
                    if (point.lightIndex < getIcNum() && point.lightIndex % 2 == 0)
                        pointsNew.push_back(newPoint);
                } else {
                    LOGE(LOG_TAG, "【补点流程A】---lightIndex 小于0   index=%d",
                         newPoint.lightIndex);
                }
            }
        }
        if (point.lightIndex >= 0) {
            lastLightIndex = point.lightIndex;//重新开始
            lastPoint = point;
            if (point.lightIndex < getIcNum() && point.lightIndex % 2 == 0)
                pointsNew.push_back(point);
        } else {
            LOGE(LOG_TAG, "3---lightIndex 小于0   index=%d", point.lightIndex);
        }
    }
    LOGW(LOG_TAG, "pointsNew = %d", pointsNew.size());
    if (pointsNew.empty())return pointsNew;
    sort(pointsNew.begin(), pointsNew.end(), compareIndex);
    double averageDistance = 0;
    int averageCnt = 0;
    //补充右侧的走向推断
    for (int i = 1; i < pointsNew.size() - 1; i++) {
        LightPoint curLPoint = pointsNew[i];
        LightPoint nextLPoint = pointsNew[i + 1];
        LightPoint lastLPoint = pointsNew[i - 1];

        if ((curLPoint.lightIndex - lastLPoint.lightIndex) == 1
            && (nextLPoint.lightIndex - curLPoint.lightIndex) == 1) {
            averageDistance += norm(curLPoint.point2f - lastLPoint.point2f);
            averageCnt += 1;
        }
        int inferredNextDiff = nextLPoint.lightIndex - curLPoint.lightIndex;
        while (inferredNextDiff > 1) {
            inferredNextDiff--;
            LightPoint inferredPoint = inferredRight(curLPoint, lastLPoint, nextLPoint, i,
                                                     pointsNew,
                                                     redSameVector,
                                                     greenSameVector);
            if (inferredPoint.errorStatus != EMPTY_POINT) {
                lastLPoint = curLPoint;
                curLPoint = inferredPoint;
            } else {
                inferredNextDiff = 0;
            }
        }
    }
    averageDistance = averageDistance / averageCnt;
    LOGW(LOG_TAG, "平均距离 averageDistance = %f  averageCnt = %d", averageDistance, averageCnt);

    // 按照y值从小到大排序
    sort(pointsNew.begin(), pointsNew.end(), compareIndex);

    vector<int> errorPointIndexVector;
    //补充左侧的走向推断
    for (int i = 1; i < pointsNew.size() - 1; i++) {
        LightPoint curLPoint = pointsNew[i];
        LightPoint nextLPoint = pointsNew[i + 1];
        LightPoint lastLPoint = pointsNew[i - 1];

        double averageDistanceThreshold = averageDistance * 2;
        bool lastContinuous = curLPoint.lightIndex == (lastLPoint.lightIndex + 1) && i > 2 &&
                              lastLPoint.lightIndex == (pointsNew[i - 2].lightIndex + 1);

        double distance4Last = norm(curLPoint.point2f - lastLPoint.point2f) /
                               max((curLPoint.lightIndex * 1.0 - lastLPoint.lightIndex), 1.0);

        bool nextContinuous =
                curLPoint.lightIndex == (nextLPoint.lightIndex - 1) && i < (pointsNew.size() - 2) &&
                nextLPoint.lightIndex == (pointsNew[i + 2].lightIndex - 1);

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
             pointsNew[errorPointIndexVector[i]].lightIndex);
        pointsNew.erase(pointsNew.begin() + errorPointIndexVector[i]);
    }

    for (int i = 1; i < pointsNew.size() - 1; i++) {
        LightPoint curLPoint = pointsNew[i];
        LightPoint nextLPoint = pointsNew[i + 1];
        LightPoint lastLPoint = pointsNew[i - 1];
        int inferredLastDiff = curLPoint.lightIndex - lastLPoint.lightIndex;
        while (inferredLastDiff > 1) {
            //下一个值没有，推断点可能位置
            inferredLastDiff--;
            LightPoint inferredPoint = inferredLeft(curLPoint, lastLPoint, nextLPoint, i,
                                                    pointsNew,
                                                    redSameVector,
                                                    greenSameVector);
            if (inferredPoint.errorStatus != EMPTY_POINT) {
                nextLPoint = curLPoint;
                curLPoint = inferredPoint;
            } else {
                inferredLastDiff = 0;
            }
        }
    }


    // 按照y值从小到大排序
    sort(pointsNew.begin(), pointsNew.end(), compareIndex);

    for (int i = 1; i < pointsNew.size(); i++) {
        LightPoint curLPoint = pointsNew[i];
        LightPoint lastLPoint = pointsNew[i - 1];

        int diff = curLPoint.lightIndex - lastLPoint.lightIndex;
        //补充tensorFlow已经识别的点
        if (diff > 1 && diff <= 4) {
            //推断点可能位置
            double diffSegmentLenX = (curLPoint.point2f.x - lastLPoint.point2f.x) / diff;
            double diffSegmentLenY = (curLPoint.point2f.y - lastLPoint.point2f.y) / diff;

            double distance4Last = norm(curLPoint.point2f - lastLPoint.point2f) / diff;
            double averageDistanceThreshold = averageDistance / 5;
            if (distance4Last < averageDistanceThreshold) {
                continue;
            }

            LOGW(LOG_TAG,
                 "【补点流程D】需要补点，差值 = %d  当前序号 = %d, 上一个序号 =%d diffSegmentLenX=%f diffSegmentLenY=%f averageDistanceThreshold = %f",
                 diff,
                 curLPoint.lightIndex, lastLPoint.lightIndex, diffSegmentLenX, diffSegmentLenY,
                 averageDistanceThreshold);
            for (int j = 0; j < diff - 1; j++) {
                int needIndex = lastLPoint.lightIndex + j + 1;
                int x = lastLPoint.point2f.x + diffSegmentLenX * (j + 1);
                int y = lastLPoint.point2f.y + diffSegmentLenY * (j + 1);

                bool isGreen = ((needIndex + 1) / 2) % 2 == 0;
                LightPoint inferredPoint;
                Point2i center = Point2i(x, y);

                double distanceMin = sqrt(
                        diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);
                if (!isGreen) {
                    inferredPoint = syncRectPoints(center, distanceMin, redSameVector);
                } else {
                    inferredPoint = syncRectPoints(center, distanceMin, greenSameVector);
                }
                if (inferredPoint.errorStatus != EMPTY_POINT) {
                    inferredPoint.lightIndex = needIndex;
                    LOGD(LOG_TAG, "【补点流程D】推断点序号：%d  xy: %d-%d", inferredPoint.lightIndex,
                         inferredPoint.point2f.x, inferredPoint.point2f.y);
                    if (inferredPoint.lightIndex >= 0) {
                        pointsNew.push_back(inferredPoint);
                        continue;
                    }
                }
                LightPoint newPoint = LightPoint(Point2f(x, y), lastPoint.contourArea);
                newPoint.lightIndex = needIndex;
                LOGW(LOG_TAG, "【补点流程D】无定位 插入数据 序号 = %d", newPoint.lightIndex);
                pointsNew.push_back(newPoint);
            }
        }
    }

    //输出一张好点图
    vector <Point2i> points;
    Mat dstCircle = src.clone();
    for (int i = 0; i < pPoints.size(); i++) {
        Point2f center = pPoints[i].point2f;
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        if (pPoints[i].errorStatus != NORMAL) {
            circle(dstCircle, center, radiusCircle2, Scalar(255, 255, 0), 3);
        } else {
            circle(dstCircle, center, radiusCircle2, Scalar(0, 0, 0), 3);
        }
    }

    for (int i = 0; i < pointsNew.size(); i++) {
        Point2f center = pointsNew[i].point2f;
        points.push_back(center);
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        circle(dstCircle, center, radiusCircle2, Scalar(0, 0, 255), 2);
        putText(dstCircle, to_string(pointsNew[i].lightIndex), center, FONT_HERSHEY_SIMPLEX,
                fontScale,
                Scalar(0, 255, 255, 240), 1);
    }
    Mat outMat = dstCircle.clone();
    int ret = getMinTrapezoid(outMat, points, trapezoid4Points);
    outMats.push_back(outMat);
    outMats.push_back(dstCircle);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    LOGE(LOG_TAG, "convexTrapezoidHull ret=%d 排序耗时：%f", ret, duration.count());
    return pointsNew;
}

LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, vector <LightPoint> &pointsNew,
                         vector <LightPoint> &redSameVector, vector <LightPoint> &greenSameVector) {
    //下一个值没有，推断点可能位置
    LOGW(LOG_TAG,
         "【补点流程B】推断[下一个]点序号 = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
         curLPoint.lightIndex + 1, curLPoint.lightIndex, lastLPoint.lightIndex,
         nextLPoint.lightIndex);
    Point2i A = lastLPoint.point2f;
    Point2i B = curLPoint.point2f;
    Point2i C = nextLPoint.point2f;
    LOGD(LOG_TAG, "【补点流程B】A = %d B = %d C = %d ", lastLPoint.lightIndex,
         curLPoint.lightIndex, nextLPoint.lightIndex);
    bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
    if (!abcHorizontal && i > 2) {
        LightPoint lastLastLPoint = pointsNew[i - 2];
        Point2i lastA = lastLastLPoint.point2f;
        abcHorizontal = isApproximatelyHorizontal(lastA, A, B);
    }
    if (abcHorizontal) {
        LightPoint inferredPoint = inferredAB2Next(lastLPoint, curLPoint, redSameVector,
                                                   greenSameVector);
        if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0) {
            LOGW(LOG_TAG, "【补点流程B】推断序号成功：%d  xy: %d-%d", inferredPoint.lightIndex,
                 inferredPoint.point2f.x, inferredPoint.point2f.y);
            pointsNew.push_back(inferredPoint);
            return inferredPoint;
        }
    }
    return LightPoint(EMPTY_POINT);
}

LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, vector <LightPoint> &pointsNew,
                        vector <LightPoint> &redSameVector, vector <LightPoint> &greenSameVector) {
    LOGW(LOG_TAG,
         "【补点流程C】推断[上一个]点序号 = %d curLPoint = %d  lastLPoint = %d nextPoint = %d",
         curLPoint.lightIndex - 1, curLPoint.lightIndex, lastLPoint.lightIndex,
         nextLPoint.lightIndex);
    Point2i A = nextLPoint.point2f;
    Point2i B = curLPoint.point2f;
    Point2i C = lastLPoint.point2f;
    bool abcHorizontal = isApproximatelyHorizontal(A, B, C);
    if (!abcHorizontal && i < pointsNew.size() - 2) {
        LightPoint nextNextLPoint = pointsNew[i + 2];
        Point2i nextA = nextNextLPoint.point2f;
        abcHorizontal = isApproximatelyHorizontal(nextA, A, B);
        LOGD(LOG_TAG, "【补点流程C】3---ABC水平,推断BC中间点 nextNextLPoint=%d",
             nextNextLPoint.lightIndex);
    }
    if (abcHorizontal) {
        LOGD(LOG_TAG, "【补点流程C】2---ABC水平,推断BC中间点");
        LightPoint inferredPoint = inferredAB2Next(nextLPoint, curLPoint, redSameVector,
                                                   greenSameVector);
        if (inferredPoint.errorStatus != EMPTY_POINT && inferredPoint.lightIndex >= 0) {
            LOGW(LOG_TAG, "【补点流程C】推断序号成功：%d  xy: %d-%d", inferredPoint.lightIndex,
                 inferredPoint.point2f.x, inferredPoint.point2f.y);
            pointsNew.push_back(inferredPoint);
            return inferredPoint;
        }
    }
    return LightPoint(EMPTY_POINT);
}


LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, vector <LightPoint> &redSameVector,
                           vector <LightPoint> &greenSameVector) {
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
    bool isGreen = ((inferredLightIndex + 1) / 2) % 2 == 0;
    LOGD(LOG_TAG,
         "当前点：%d 推断点：%d  diff : %d  diffSegmentLenX : %f  diffSegmentLenY : %f isGreen=%d",
         B.lightIndex, inferredLightIndex, diff, diffSegmentLenX, diffSegmentLenY, isGreen ? 1 : 0);

    LightPoint inferredPoint;
    Point2i center = Point2i(x, y);
    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);
    if (!isGreen) {
        inferredPoint = syncRectPoints(center, distanceMin, redSameVector);
    } else {
        inferredPoint = syncRectPoints(center, distanceMin, greenSameVector);
    }
    if (inferredPoint.errorStatus != EMPTY_POINT) {
        inferredPoint.lightIndex = inferredLightIndex;
    }
    return inferredPoint;
}

LightPoint inferredAB2C(LightPoint &A, LightPoint &B, vector <LightPoint> &redSameVector,
                        vector <LightPoint> &greenSameVector) {
    int diff;
    if (A.lightIndex < 0) {
        diff = getLightIndex(A.score) - B.lightIndex - 1;
    } else {
        diff = A.lightIndex - B.lightIndex - 1;
    }
    double diffSegmentLenX =
            (A.point2f.x - B.point2f.x) / (diff + 1);
    double diffSegmentLenY =
            (A.point2f.y - B.point2f.y) / (diff + 1);

    int curLightIndex, x, y;
    if (diff > 0) {
        curLightIndex = B.lightIndex - 1;
        x = B.point2f.x - diffSegmentLenX;
        y = B.point2f.y - diffSegmentLenY;
    } else {
        curLightIndex = B.lightIndex + 1;
        x = B.point2f.x + diffSegmentLenX;
        y = B.point2f.y + diffSegmentLenY;
    }
    bool isGreen = ((curLightIndex + 1) / 2) % 2 == 0;
    LOGD(LOG_TAG, "当前点：%d 推断点：%d  diff : %d  diffSegmentLenX : %f  diffSegmentLenY : %f ",
         B.lightIndex, curLightIndex, diff, diffSegmentLenX, diffSegmentLenY);

    LightPoint inferredPoint;
    Point2i center = Point2i(x, y);
    LOGD(LOG_TAG, "A：%d(%d)-%d-%d B：%d-%d-%d  inferredPoint: %d-%d-%d ", A.score, A.lightIndex,
         A.point2f.x,
         A.point2f.y, B.lightIndex, B.point2f.x,
         B.point2f.y, curLightIndex, center.x, center.y);
    double distanceMin = sqrt(
            diffSegmentLenX * diffSegmentLenX + diffSegmentLenY * diffSegmentLenY);
    if (!isGreen) {
        inferredPoint = syncRectPoints(center, distanceMin, redSameVector);
    } else {
        inferredPoint = syncRectPoints(center, distanceMin, greenSameVector);
    }
    if (inferredPoint.errorStatus != EMPTY_POINT) {
        inferredPoint.lightIndex = curLightIndex;
    }
    return inferredPoint;
}


/**
 * 对齐图片
 */
Mat alignImg(Mat & src, Mat & trans, bool
back4Matrix) {
if (src.

empty()

) {
LOGE(LOG_TAG, " src empty ");
}
if (trans.

empty()

) {
LOGE(LOG_TAG, " trans empty ");
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
cv::Mat mask = cv::Mat::zeros(trans.size(), CV_8UC1);
Rect searchRegion;
searchRegion = Rect(pointsAreaLeft, pointsAreaTop, pointsAreaRight - pointsAreaLeft,
                    pointsAreaBottom - pointsAreaTop);
int area = searchRegion.width * searchRegion.height;
if (area < 25) {
// 假设我们只想在目标图像的一个特定区域内搜索
LOGE(LOG_TAG, "area < 25,use hard rect");
searchRegion = Rect(120, 120, 400, 400); // x, y, width, height
}

rectangle(mask, searchRegion, Scalar::all(255), FILLED
);

// Convert images to grayscale
Mat alignedImg;
Mat im1Src, im2Trans;
//    //转换为灰度图
cvtColor(src, im1Src, CV_BGR2GRAY
);
cvtColor(trans, im2Trans, CV_BGR2GRAY
);
TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
                      number_of_iterations2, termination_eps2);
findTransformECC(im1Src, im2Trans, warp_matrix,
        motionTypeSet, criteria, mask
);
if (motionTypeSet == MOTION_HOMOGRAPHY) {
warpPerspective(trans, alignedImg, warp_matrix, trans
.

size(), INTER_LINEAR

+ WARP_INVERSE_MAP);
} else {
warpAffine(trans, alignedImg, warp_matrix, trans
.

size(), INTER_LINEAR

+ WARP_INVERSE_MAP);
}
if (back4Matrix) {
return
warp_matrix;
}
//    // 在图像上绘制矩形
//    rectangle(alignedImg, searchRegion, Scalar(255, 255, 0), 2);
return
alignedImg;

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
vector <LightPoint>
        findColorType(Mat & src, int
stepFrame,
vector <LightPoint> &points,
        vector<Mat>
&outMats) {
vector <LightPoint> result;
Mat meanColorMat = src.clone();
for (
int i = 0;
i<points.

size();

i++) {
LightPoint lPoint = points[i];
Scalar scalar;
LightPoint lightPoint = meanColor(src, stepFrame, lPoint, meanColorMat);
result.
push_back(lightPoint);
}
outMats.
push_back(meanColorMat);
return
result;
}

/**
 * 获取区域 hsv 色相
 */
LightPoint meanColor(Mat & src, int
stepFrame,
LightPoint &lPoint, Mat
&meanColorMat) {
if (src.

empty()

) {
LOGE(LOG_TAG, "meanColor(stepFrame=%d): Error: Image not found!", stepFrame);
return

LightPoint();

}
Point2f point = lPoint.point2f;
int x = point.x; // 指定坐标x
int y = point.y; // 指定坐标y
int radius = 5; // 指定半径

Rect roi(x - radius, y - radius, 2 * radius, 2 * radius);

if (x - radius < 0) {
roi.
x = 0;
LOGE(LOG_TAG, "x<0");
}
if (y - radius < 0) {
roi.
y = 0;
LOGE(LOG_TAG, "y<0");
}
if (roi.x + roi.width > src.cols) {
LOGE(LOG_TAG, "x>cols with:%d  src-cols: %d", (roi.x + roi.width), src.cols);
roi.
width = src.cols - roi.x;
}
if (roi.y + roi.height > src.rows) {
LOGE(LOG_TAG, "y>rows height:%d  src-rows: %d", (roi.y + roi.height), src.rows);
roi.
height = src.rows - roi.y;
}

Mat region = src(roi);
Scalar avgPixelIntensity = mean(region);

//    double blue = avgPixelIntensity[0];
double green = avgPixelIntensity[1];
double red = avgPixelIntensity[2];

Mat hsv;
cvtColor(region, hsv, COLOR_BGR2HSV
);
Scalar mean = cv::mean(hsv);

CUS_COLOR_TYPE colorType = E_W;
circle(meanColorMat, point, radius, Scalar(255, 255, 255),
0.7);
if (red > green) {//red > blue &&
colorType = E_RED;
putText(meanColorMat,
"red", point, FONT_HERSHEY_SIMPLEX, 0.5,
Scalar(255, 255, 255), 1);

} else if (green > red) {// && green > blue
colorType = E_GREEN;
putText(meanColorMat,
"green", point, FONT_HERSHEY_SIMPLEX, 0.5,
Scalar(255, 255, 255), 1);
} else {
LOGV(LOG_TAG, "meanColor= 无法识别");
putText(meanColorMat,
"UnKnow", point, FONT_HERSHEY_SIMPLEX, 0.5,
Scalar(255, 255, 255), 1);
}

return lPoint.
copyPoint(colorType, mean
);
}

bool isApproximatelyHorizontal(Point2i A, Point2i B, Point2i C) {
    double slopeAB = (double) (B.y - A.y) / (B.x - A.x);
    double slopeBC = (double) (C.y - B.y) / (C.x - B.x);

    // 定义一个阈值，用于判断斜率是否接近水平线的斜率
    double threshold = 0.45;
//    LOGD(LOG_TAG, "slopeAB = %f  slopeBC = %f  threshold = %f", slopeAB, slopeBC,
//         abs(slopeAB - slopeBC));
    if (abs(slopeAB - slopeBC) < threshold) {
        return true;
    } else {
        return false;
    }
}

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector <LightPoint> &points) {
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
string point2iToJson(const vector <Point2i> &points) {
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
        ss << "\"pName\": " << name << ", ";
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

//todo:应该时更偏向线性方向优先
LightPoint syncRectPoints(Point2i &center, double minDistance,
                          vector <LightPoint> &points) {
    if (minDistance > 150) {
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
        LOGE(LOG_TAG, "找不到推断点");
        return LightPoint(EMPTY_POINT);
    }
    LightPoint selectPoint = points[selectIndex];
    points.erase(points.begin() + selectIndex);
//    LOGD(LOG_TAG, "points剩余=%d  x-y: %d- %d", points.size(), selectPoint.point2f.x,
//         selectPoint.point2f.x);
    return selectPoint;
}
