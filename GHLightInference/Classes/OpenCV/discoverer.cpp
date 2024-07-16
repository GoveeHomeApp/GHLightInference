#include "discoverer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
//#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
/**
 * 查找灯带光点
 */
void
findByContours(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints, int icNum,
               vector<Mat> &outMats) {
    LOGD(LOG_TAG, "=====查找灯带光点");
    // 将图像从BGR转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
//
//    // 分离HSV通道
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
//    // 定义增强系数
    double sFactor = 1.2; // 饱和度增强系数
//    double vFactor = 1.3; // 亮度增强系数
//
//    // 创建S和V通道的LUT
    int lutSize = 256;
    cv::Mat sLUT(1, lutSize, CV_8UC1);
//    cv::Mat vLUT(1, lutSize, CV_8UC1);
    for (int i = 0; i < lutSize; ++i) {
        // 增强饱和度
        sLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * sFactor);
        // 增强亮度
        // vLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * vFactor);
    }

    // 应用LUT到S通道
    cv::LUT(hsvChannels[1], sLUT, hsvChannels[1]);
    // 应用LUT到V通道
    //  cv::LUT(hsvChannels[2], vLUT, hsvChannels[2]);
    // 合并增强后的通道回HSV图像
    cv::merge(hsvChannels, hsv);
    // 将HSV图像转换回BGR颜色空间以便显示
    cv::Mat enhanced, enhancedGay;
    cv::cvtColor(hsv, enhanced, cv::COLOR_HSV2BGR);
    outMats.push_back(enhanced);

    vector<Mat> singleImgs;
    split(enhanced, singleImgs);
    //BGR
    Mat rImg = singleImgs[2];
    Mat gImg = singleImgs[1];
    Mat gray;
    cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);
    Mat dst = thresholdPoints(gImg, image, hsvChannels[0], 1, outMats);
    Mat dst2 = thresholdPoints(rImg, image, hsvChannels[0], 2, outMats);

    // 寻找轮廓
    vector<vector<Point2f>> contoursA;
    vector<vector<Point2f>> contoursB;
    findContours(dst, contoursA, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(dst2, contoursB, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contoursA.size(); i++) {
        vector<Point2f> contour = contoursA[i];
        double contourArea = cv::contourArea(contour);
        if (contourArea < 1.0)continue;
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        pointVector.push_back(Point2f(center.x, center.y));
        LightPoint lp = LightPoint();
        lp.position = center;
        lightPoints.push_back(lp);
        LOGV("contourArea", "drawColorMask contourArea: %f  radius: %f", contourArea, radius);
    }
    for (int i = 0; i < contoursB.size(); i++) {
        vector<Point2f> contour = contoursB[i];
        double contourArea = cv::contourArea(contour);
        if (contourArea < 1.0)continue;
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        pointVector.push_back(Point2f(center.x, center.y));
        LightPoint lp = LightPoint();
        lp.position = center;
        lightPoints.push_back(lp);
        LOGV("contourArea", "drawColorMask contourArea: %f  radius: %f", contourArea, radius);
    }
    LOGD(LOG_TAG, "合并点位前 pointVector =%d ", pointVector.size());
    mergePoints(pointVector, 3);

    LOGD(LOG_TAG, "合并点位后 pointVector =%d ", pointVector.size());

    Mat outMat2 = image.clone();
    polyPoints(pointVector, 3, 2.3, outMat2);
    outMats.push_back(outMat2);
}

void findNoodleLamp(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints,
                    vector<Mat> &outMats) {
    LOGD(LOG_TAG, "tf识别数量：%d", lightPoints.size());
    Mat tfOut = image.clone();
    for (int i = 0; i < lightPoints.size(); i++) {
        LightPoint lp = lightPoints[i];
        Rect2i tfRect = lp.tfRect;
        double height = max(tfRect.width, tfRect.height);
        if (height < 300) {
            for (int k = 0; k < 4; k++) {
                //originalFrame1C, roi, Scalar(255, 0, 50), 4);
                rectangle(tfOut, tfRect, Scalar(255, 0, 50), 2);
            }
        }
    }

    putText(tfOut, "tensorFRect", Point(50, 50), FONT_HERSHEY_SIMPLEX,
            0.7, Scalar(255, 0, 50), 2);
    outMats.push_back(tfOut);


    try {
        Mat dst = thresholdNoodleLamp(image, lightPoints, outMats);
    } catch (...) {
        LOGE(LOG_TAG, "异常状态30");
    }
}

vector<int>
polyPoints(vector<Point2f> &pointVector, int k, double stddevThreshold, Mat &outMat) {
    vector<int> eraseVector;
    if (pointVector.empty()) {
        LOGE(LOG_TAG, "polyPoints null");
        return eraseVector;
    }
    if (pointVector.size() < k + 4)
        return eraseVector;
    try {
        Mat pointsMat(pointVector);
        pointsMat.convertTo(pointsMat, CV_32F);

        LOGD(LOG_TAG, "pointVector: %d", pointVector.size());
        Mat labels, centers;
        kmeans(pointsMat, k, labels,
               TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 12, 0.1),
               3, KMEANS_PP_CENTERS, centers);

        // 计算每个数据点到其对应聚类中心的距离
        vector<float> distances;
        map<int, vector<float>> distancesMap;
        for (int i = 0; i < pointsMat.rows; i++) {
            Point2f point = pointsMat.at<Point2f>(i);
            int type = labels.at<int>(i);
            Point2f center = centers.at<Point2f>(type);
            float distance = norm(point - center);
            distances.push_back(distance);
        }
        // 计算离群点的阈值
        Scalar mean, stddev;
        meanStdDev(distances, mean, stddev);
        float threshold = mean[0] + stddevThreshold * stddev[0];
        int size = pointVector.size();

        // 输出离群点
        for (int i = pointsMat.rows - 1; i >= 0; i--) {
            if (distances[i] > threshold) {
                circle(outMat, pointVector[i], 10, Scalar(0, 0, 0), 2);
                if (pointVector.begin() + i < pointVector.end()) {
                    pointVector.erase(pointVector.begin() + i);
                    eraseVector.push_back(i);
                }
            }
        }

        LOGD(LOG_TAG, "pointVector擦除离群点 = %d", size - pointVector.size());
        vector<vec4f> data;
        for (int i = 0; i < pointVector.size(); i++) {
            if (i < labels.rows && labels.at<int>(i) == 0) {
                circle(outMat, pointVector[i], 7, Scalar(0, 0, 255, 160), 2);
            } else if (i < labels.rows && labels.at<int>(i) == 1) {
                circle(outMat, pointVector[i], 7, Scalar(0, 255, 0, 160), 2);
            } else {
                circle(outMat, pointVector[i], 7, Scalar(255, 0, 0, 160), 2);
            }
            data.push_back(vec4f{pointVector[i].x * 1.f, pointVector[i].y * 1.f});
        }
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常5");
    }
    return eraseVector;
}

// 计算两点之间的距离
double distance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算线段的角度
double angle(Vec4i line) {
    return atan2(line[3] - line[1], line[2] - line[0]);
}

double distanceBetweenPoints(Point2f p1, Point2f p2) {
    Point2f diff = p1 - p2;
    return sqrt(diff.x * diff.x + diff.y * diff.y);
}

double calculateUniformIntervalScore(const vector<LightBead> &beads) {
    if (beads.size() < 3) return 1.0;  // 如果灯珠数量少于3，认为是均匀的

    vector<double> intervals;
    for (size_t i = 1; i < beads.size(); i++) {
        double interval = norm(beads[i].center - beads[i - 1].center);
        intervals.push_back(interval);
    }

    double meanInterval = accumulate(intervals.begin(), intervals.end(), 0.0) / intervals.size();
    double maxDeviation = 0.0;

    for (const auto &interval: intervals) {
        double deviation = abs(interval - meanInterval) / meanInterval;
        maxDeviation = max(maxDeviation, deviation);
    }

    return 1.0 - min(maxDeviation, 1.0);  // 归一化到 [0, 1] 范围
}

bool isBeadAligned(const vector<LightBead> &bar, const LightBead &bead) {
    if (bar.size() < 2) return true;  // 如果灯条中还不足两个灯珠，任何新灯珠都可以加入
    // 检查新灯珠是否与灯条末端足够接近，但不超过80
    RotatedRect rect = fitRotatedRect(bar);
    float radius = 0.0f;
    if (!bar.empty()) {
        for (const auto &item: bar) {
            radius += item.radius;
        }
        radius = radius / bar.size();
    }
    radius = max(radius, 4.0f);

    // 检查是否所有轮廓都大致在一条直线上
    Vec4f line;
    vector<Point2f> centers;
    for (const auto &c: bar) {
        centers.push_back(c.center);
    }

    fitLine(centers, line, DIST_L2, 0, 0.01, 0.01);

    int allAligned = 1;
    double maxDistance = 0;
    for (const auto &c: bar) {
        Point vec(line[0], line[1]);
        Point ptOnLine(line[2], line[3]);
        Point ptToCenter = c.center - ptOnLine;
        float distance = abs(ptToCenter.cross(vec));
        if (distance > max(30.0f, radius)) {  // 允许的最大偏差
            LOGW(LOG_TAG, "允许的最大偏差");
            allAligned = 0;
            break;
        }
        maxDistance = max(maxDistance,
                          distanceBetweenPoints(c.center, bar[0].center));
    }
    // 检查间距
    bool hasOnline = (allAligned == 1) && maxDistance <= radius * (bar.size() - 1) * 4;
    LOGW(LOG_TAG, "maxDistance=%f,  maxL=%f allAligned=%d radius=%f  barSize=%d", maxDistance,
         (radius * (bar.size() - 1) * 4), allAligned, radius, bar.size());
    float newLength = min(distanceP(bead.center, bar.back().center),
                          distanceP(bead.center, bar.front().center));
//    // 检查新灯珠是否在灯条的延长线上
//    float dotProduct = direction.dot(newDirection / newLength);
//    float angleThreshold = cos(CV_PI / 18);  // 允许10度的角度偏差

    float min = cv::max(cv::min(rect.size.width, rect.size.height), radius);
    float distanceThreshold = max(min * 4.0f, 35.0f);
    LOGD(LOG_TAG,
         "distanceThreshold = %f    radius = %f     newLength = %f  ", distanceThreshold, radius,
         newLength);
    return hasOnline && newLength < distanceThreshold;
}

RotatedRect fitRotatedRect(const vector<LightBead> &beads) {
    vector<Point2f> points;
    for (const auto &bead: beads) {
        points.push_back(bead.center);
    }
    return minAreaRect(points);
}


Rect2i safeRect(const cv::Rect2i &region, const cv::Size &imageSize) {
    cv::Rect2i safe = region;
    safe.x = safe.x - 4;
    safe.y = safe.y - 4;
    safe.width = safe.width + 4;
    safe.height = safe.height + 4;
    safe.x = std::max(0, std::min(safe.x, imageSize.width - 1));
    safe.y = std::max(0, std::min(safe.y, imageSize.height - 1));
    safe.width = std::min(safe.width, imageSize.width - safe.x);
    safe.height = std::min(safe.height, imageSize.height - safe.y);
    return safe;
}

/**
 计算 region 的对角线长度，作为延伸的最大长度。
从灯带的中心点开始，向两个方向延伸，每个方向延伸 region 对角线长度的一半。
使用 lambda 函数 clipToRegion 确保延伸后的端点不会超出 region 的边界。
基于裁剪后的端点计算新的中心点和长度
 */
cv::RotatedRect extendRotatedRect(const cv::RotatedRect &rect, const cv::Rect &region) {
    // 计算矩形的方向向量
    float angle = rect.angle * CV_PI / 180.0;
    cv::Point2f direction(std::cos(angle), std::sin(angle));

    // 计算region的对角线长度
    float regionLength = std::sqrt(region.width * region.width + region.height * region.height);

    // 计算延伸后的端点
    cv::Point2f center = rect.center;
    cv::Point2f extended1 = center + direction * (regionLength / 2);
    cv::Point2f extended2 = center - direction * (regionLength / 2);

    // 将端点裁剪到区域内
    auto clipToRegion = [&region](cv::Point2f &p) {
        p.x = std::max(float(region.x), std::min(p.x, float(region.x + region.width - 1)));
        p.y = std::max(float(region.y), std::min(p.y, float(region.y + region.height - 1)));
    };

    clipToRegion(extended1);
    clipToRegion(extended2);

    // 计算新的中心点和大小
    cv::Point2f newCenter = (extended1 + extended2) * 0.5f;
    float newLength = cv::norm(extended1 - extended2);

//    float angleR = rect.angle;
//    // 保持原始宽度不变
//    if (rect.size.width > rect.size.height && newLength < rect.size.height) {
//        angleR += 90.0f;
//        if (angleR > 180.0f) {
//            angleR -= 180.0f;
//        }
//    }

    return cv::RotatedRect(newCenter, cv::Size2f(newLength, rect.size.height), rect.angle);
}

RotatedRect findLightStripInRegion(const Mat &input, const Rect2i &region, vector<Mat> &outMats) {
    Rect2i safeRegion = safeRect(region, input.size());
    if (safeRegion.width <= 0 || safeRegion.height <= 0) {
        return cv::RotatedRect(); // 返回一个空的RotatedRect
    }

    cv::Mat roi = input(safeRegion);

    Mat gray, enhanced, blurred, binary, morphed;

    // 转换为灰度图
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    // 增强亮度对比度
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(gray, enhanced);

    // 应用高斯模糊以减少噪声
    GaussianBlur(enhanced, blurred, Size(1, 1), 0);


    vector<vector<Point>> contours;
    int threshValue = 180;
    double regionArea = safeRegion.width * safeRegion.height * 0.3;
//    LOGD(LOG_TAG, "regionArea = %f", regionArea);
    while (threshValue < 220) {
        threshold(blurred, binary, threshValue, 255, THRESH_BINARY);
        binary = morphologyImage(binary, 3, 0, MORPH_ELLIPSE);
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // 找到面积最大的轮廓
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
            }
        }
        if (maxArea < regionArea) {
            break;
        }
        threshValue += 6;
        LOGD(LOG_TAG, "contours = %d   thresh = %d", contours.size(), threshValue);
    }
    contours = removeLargeContours(contours);

//    outMats.push_back(binary);
    // 合并所有轮廓的点集
    vector<Point> allPoints;
    for (const auto &contour: contours) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    if (allPoints.empty()) {
        LOGE(LOG_TAG, "allPoints empty !");
        return RotatedRect();
    }
    // 计算最小包裹矩形
    RotatedRect boundingBox = minAreaRect(allPoints);
    if (boundingBox.size.area() > 0) {
        // 调整坐标到原图
        boundingBox.center.x += safeRegion.x;
        boundingBox.center.y += safeRegion.y;
        // 将RotatedRect的坐标转换回原图坐标系
//        boundingBox = extendRotatedRect(boundingBox, safeRegion);
    }
    return boundingBox;
}

std::vector<std::vector<cv::Point>>
removeLargeContours(const std::vector<std::vector<cv::Point>> &contours, double threshold ,
                    size_t minContourCount) {
    // 如果轮廓数量不大于minContourCount，直接返回原始轮廓
    if (contours.size() <= minContourCount) {
        return contours;
    }

    // 计算所有轮廓的面积
    std::vector<double> areas;
    for (const auto &contour: contours) {
        areas.push_back(cv::contourArea(contour));
    }

    // 计算面积的中位数
    size_t n = areas.size() / 2;
    std::nth_element(areas.begin(), areas.begin() + n, areas.end());
    double median_area = areas[n];

    // 如果是偶数个元素，取中间两个数的平均
    if (areas.size() % 2 == 0) {
        std::nth_element(areas.begin(), areas.begin() + n - 1, areas.end());
        median_area = (median_area + areas[n - 1]) / 2.0;
    }

    // 过滤掉面积大于阈值倍中位数的轮廓
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto &contour: contours) {
        if (cv::contourArea(contour) <= threshold * median_area) {
            filtered_contours.push_back(contour);
        }
    }

    return filtered_contours;
}

vector<LightPoint> removeLargeRectangles(vector<LightPoint> &lightStrips) {
    // 步骤 1: 计算平均长度
    vector<double> lengths;
    for (const auto &lp: lightStrips) {
        int length = max(lp.tfRect.width, lp.tfRect.height);
        if (length <= 450) {
            lengths.push_back(length);
        }
    }

    // 如果没有符合条件的矩形，直接返回原始向量
    if (lengths.empty()) {
        for (const auto &lp: lightStrips) {
            int length = max(lp.tfRect.width, lp.tfRect.height);
            lengths.push_back(length);
        }
    }

    // 步骤 2: 排除最大值和最小值
    if (lengths.size() > 2) {
        sort(lengths.begin(), lengths.end());
        lengths.erase(lengths.begin());
        lengths.pop_back();
    }

    // 步骤 3: 计算平均长度
    double avgLength = accumulate(lengths.begin(), lengths.end(), 0.0) / lengths.size();

    // 步骤 4: 移除大于平均长度 1.5 倍的矩形
    vector<LightPoint> result;
    for (const auto &lp: lightStrips) {
        int length = max(lp.tfRect.width, lp.tfRect.height);
        if (length <= avgLength * 1.5) {
            result.push_back(lp);
        }
    }
    return result;
}

vector<RotatedRect>
findLightStrips(const Mat &input, vector<LightPoint> &lps, vector<Mat> &outMats) {
    vector<RotatedRect> lightStrips;
    try {
        int totalLength = 0;
        vector<LightPoint> results = removeLargeRectangles(lps);
        if (lps.empty())return lightStrips;

        for (const auto &lightPoint: results) {
            Rect2i region = lightPoint.tfRect;
            LOGD(LOG_TAG, "tfRect %d - %d  wh %d - %d", region.x, region.y, region.width,
                 region.height);
            int height = max(region.width, region.height);
            totalLength += height;
            RotatedRect strip = findLightStripInRegion(input, region, outMats);
            if (strip.size.area() > 0) {  // 检查是否找到了有效的灯带
                lightStrips.push_back(strip);
            }
        }
    } catch (...) {
        LOGE(LOG_TAG, "findLightStrips error");
    }
    LOGD(LOG_TAG, "lightStrips = %f", lightStrips.size());
    return lightStrips;
}

Mat thresholdNoodleLamp(Mat &image, vector<LightPoint> &lightPoints, vector<Mat> &outMats) {
    vector<RotatedRect> rRectList = findLightStrips(image, lightPoints, outMats);
    LOGD(LOG_TAG, "rRectList = %d", rRectList.size());
    lightPoints.clear();
    try {
        LOGE(LOG_TAG, "rRectList = %d ", rRectList.size());
        for (int i = 0; i < rRectList.size(); i++) {
            RotatedRect resultRect = rRectList[i];
            LightPoint lp = LightPoint();
            lp.with = resultRect.size.width;
            lp.height = resultRect.size.height;
            lp.tfRect = resultRect.boundingRect();
            lp.rotatedRect = resultRect;
            lp.position = resultRect.center;
            lightPoints.push_back(lp);
        }
    } catch (...) {
        LOGE(LOG_TAG, "异常状态29");
    }
    LOGW(LOG_TAG, "矩形轮廓数量lightPoints= %d", lightPoints.size());
    return image;
}

Mat thresholdPoints(Mat &src, Mat &bgrSrc, Mat &hue, int color,
                    vector<Mat> &outMats) {
    Mat morphology_image, threshold_image;
    int contoursSize = 1200;
    int thresh = 200;
    // 寻找轮廓
    vector<vector<Point>> contours;
//    outMats.push_back(src);
    while (contoursSize > 165 && thresh < 225) {
        threshold(src, threshold_image, thresh, 255, THRESH_BINARY);
        morphology_image = morphologyImage(threshold_image, 5, 7, MORPH_ELLIPSE);
        findContours(morphology_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        contoursSize = contours.size();
        thresh += 5;
        LOGD(LOG_TAG, "contours = %d   thresh = %d", contours.size(), thresh);
    }
//    outMats.push_back(morphology_image);

    sort(contours.begin(), contours.end(), compareContourAreas);

    // 定义腐蚀的结构元素
    Mat kernelErode = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    // 对面积大于200的轮廓进行腐蚀操作
    Mat dst = cv::Mat::zeros(morphology_image.size(), morphology_image.type());

    bool hasErode = false;

    // 对大于200像素的轮廓区域进行腐蚀操作
    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓面积
        double contourArea = cv::contourArea(contours[i]);
        Mat mask = Mat::zeros(hue.size(), CV_8UC1);

        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);
        radius = radius > 9.0 ? radius : 9.0;
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        circle(mask, center, radius, Scalar(255), -1);

        drawContours(mask, contours, i, Scalar(255), FILLED);
        Scalar mean, stddev;
        meanStdDev(hue, mean, stddev, mask);
        LOGV(LOG_TAG, " contourArea= %f  mean= %f stddev= %f  ", contourArea, mean[0], stddev[0]);

        if (contourArea > 400 || stddev[0] < 2) {
            LOGV(LOG_TAG, "超出范围或者方差太小 contourArea= %f stddev= %f", contourArea,
                 stddev[0]);
        } else if (contourArea >= 230) {
//            if (stddev[0] > 12) {
            // 对该轮廓区域进行腐蚀操作
            cv::drawContours(dst, contours, static_cast<int>(i), cv::Scalar::all(255), -1);
//            }
        } else {
            if (!hasErode) {
//                outMats.push_back(dst);
                hasErode = true;
                Mat kernelErodeMin = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
//                erode(dst, dst, kernelErode, Point2f(-1, -1), 1);
                erode(dst, dst, kernelErodeMin);
                outMats.push_back(dst);
            }
            cv::drawContours(dst, contours, static_cast<int>(i), cv::Scalar::all(255), -1);
        }
    }
    outMats.push_back(dst);
    return dst;
}

double calculateDistance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void mergePoints(vector<Point2f> &points, double threshold) {
    try {
        vector<Point2f> mergedPoints;
        vector<bool> merged(points.size(), false);

        for (int i = 0; i < points.size(); i++) {
            if (merged[i]) continue;

            Point2f mergedPoint = points[i];
            int count = 1;

            for (int j = i + 1; j < points.size(); j++) {
                if (!merged[j] && calculateDistance(points[i], points[j]) < threshold) {
                    mergedPoint += points[j];
                    count++;
                    merged[j] = true;
                }
            }

            mergedPoint /= count;
            mergedPoints.push_back(mergedPoint);
            points = mergedPoints;
        }
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常6");
    }
}

bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
    double area1 = contourArea(contour1);
    double area2 = contourArea(contour2);
    return (area1 > area2);
}

/**
* opencvMorphShapes的类型是一个枚举类型，包含以下几种形状：
1. MORPH_RECT：矩形形状
2. MORPH_CROSS：十字形状
3. MORPH_ELLIPSE：椭圆形状
参数
1. shape：结构元素的形状，可以是MORPH_RECT（矩形）、MORPH_CROSS（十字形）、MORPH_ELLIPSE（椭圆形）等。
2. size：结构元素的尺寸，通常为奇数，表示结构元素的宽度和高度。
3. anchor：锚点位置，指定结构元素的中心位置，默认为(-1, -1)，表示结构元素的中心。

1. MORPH_ERODE：腐蚀操作，用于缩小物体的边界。
2. MORPH_DILATE：膨胀操作，用于扩大物体的边界。
3. MORPH_OPEN：开运算，先腐蚀后膨胀，用于去除小的噪点。
4. MORPH_CLOSE：闭运算，先膨胀后腐蚀，用于填充物体内部的小洞。
*/
Mat morphologyImage(Mat &image, int openKernelSize,
                    int dilateKernelSize, int shape
) {
    Mat outMat;
    Mat openKernel = getStructuringElement(shape, Size(openKernelSize, openKernelSize));

    Mat morphologyImage;
    morphologyEx(image, outMat, MORPH_OPEN, openKernel);
    if (dilateKernelSize > 0) {
        Mat dilateKernel = getStructuringElement(shape,
                                                 Size(dilateKernelSize, dilateKernelSize));
        morphologyEx(outMat, outMat, MORPH_DILATE, dilateKernel, Point2f(-1, -1),
                     1);
    }
    return outMat;
}

// 计算两点之间的距离
double distanceP(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算包含所有点的最小等腰梯形
int
getMinTrapezoid(Mat &image, const vector<Point2f> &pointsSrc, vector<Point2f> &trapezoid4Points) {
    if (pointsSrc.empty() || image.empty()) {
        LOGE(LOG_TAG, "getMinTrapezoid null");
        return 0;
    }
    try {
        vector<Point2f> points(pointsSrc);
        polyPoints(points,
                   3, 1.9, image);
        vector<Point2f> hull;
        convexHull(points, hull);
        vector<double> angleVector;
        vector<bool> rightVector;
        //左右凸包最最靠边角点
        Point2f pointRight(0, 0), pointLeft(0, 0);
        // 计算凸包的中心点
        Moments mu = moments(hull);
        for (int i = 0; i < hull.size(); i++) {
            Point2f point1 = hull[i];
            Point2f point2 = hull[(i + 1) % hull.size()];
            // 计算两个点的连线的斜率
            double slope = (double) (point2.y - point1.y) / (double) (point2.x - point1.x);
            // 计算斜率与水平方向的夹角
            double angle = atan(slope) * 180 / CV_PI;
            angleVector.push_back(angle);
            rightVector.push_back(point1.x > mu.m10 / mu.m00);
            putText(image, to_string(angle), point1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
            if (pointLeft.x > point1.x || pointLeft.x == 0) {
                pointLeft = point1;
            }
            if (pointRight.x < point1.x || pointRight.x == 0) {
                pointRight = point1;
            }
        }
        line(image, Point2f(mu.m10 / mu.m00, 0), Point2f(mu.m10 / mu.m00, image.rows),
             Scalar(255, 255, 255), 2);

        double averageSlope = 75;

        //计算最接近均值斜率的斜线
        double closestAngleRight;
        double closestAngleLeft;
        double angleDiffFLagRight = 0;
        double angleDiffFLagLeft = 0;
        for (int i = 0; i < angleVector.size(); i++) {
            double angle = angleVector[i];
            if (abs(angle) < 40) {
                continue;
            }
            if (rightVector[i]) {//右侧
                if (angle < 0)continue;
                if (angleDiffFLagRight == 0) {
                    angleDiffFLagRight = abs(abs(angle) - averageSlope);
                    closestAngleRight = angle;
                } else {
                    double curDiff = abs(abs(angle) - averageSlope);
                    if (angleDiffFLagRight > curDiff) {
                        //当前角度更接近
                        closestAngleRight = angle;
                        angleDiffFLagRight = curDiff;
                    }
                }
            } else {//左侧
                if (angle > 0)continue;
                if (angleDiffFLagLeft == 0) {
                    angleDiffFLagLeft = abs(abs(angle) - averageSlope);
                    closestAngleLeft = angle;
                } else {
                    double curDiff = abs(abs(angle) - averageSlope);
                    if (angleDiffFLagLeft > curDiff) {
                        //当前角度更接近
                        closestAngleLeft = angle;
                        angleDiffFLagLeft = curDiff;
                    }
                }
            }
        }

        double angleSelect = 0;
        if (abs(pointRight.x - mu.m10 / mu.m00) > abs(mu.m10 / mu.m00 - pointLeft.x)) {
            pointRight.x = pointRight.x - 5;
            //取右边点
            int leftX = mu.m10 / mu.m00 - (pointRight.x - mu.m10 / mu.m00);
            pointLeft = Point2f(leftX, pointRight.y);
        } else {
            pointLeft.x = pointLeft.x + 5;
            int rightX = mu.m10 / mu.m00 + (mu.m10 / mu.m00 - pointLeft.x);
            pointRight = Point2f(rightX, pointLeft.y);
        }
        if (abs(closestAngleRight) > abs(closestAngleLeft)) {
            angleSelect = abs(closestAngleRight);
        } else {
            angleSelect = abs(closestAngleLeft);
        }

        LOGD(LOG_TAG, "均值斜率：%f", averageSlope);

        if (angleSelect <= 5) {
            LOGE(LOG_TAG, "左右均无有效斜边");
            return 0;
        } else if (angleSelect > 77) {
            angleSelect = 77;
        }

        circle(image, pointRight,
               10, Scalar(0, 0, 0), 5);
        circle(image, pointLeft,
               10, Scalar(0, 0, 0), 5);
        int minY = hull[0].y;
        for (int i = 1; i < hull.size(); i++) {
            if (hull[i].y < minY) {
                minY = hull[i].y;
            }
        }
        // 计算AB连线的斜率
        double slopeLeft = tan(-angleSelect * CV_PI / 180);
        double slopeRight = tan(angleSelect * CV_PI / 180);

        // 计算A点的x轴坐标
        double leftTopX = pointLeft.x - (pointLeft.y - minY) / slopeLeft;
        double rightTopX = pointRight.x - (pointRight.y - minY) / slopeRight;

        line(image, pointLeft, Point2f(leftTopX, minY),
             Scalar(0, 0, 255),
             3);

        line(image, pointRight, Point2f(rightTopX, minY),
             Scalar(0, 0, 255), 3);
        trapezoid4Points.push_back(Point2f(rightTopX, minY));
        trapezoid4Points.push_back(pointRight);
        trapezoid4Points.push_back(pointLeft);
        trapezoid4Points.push_back(Point2f(leftTopX, minY));

        LOGD(LOG_TAG, "closestAngleRight：%f   closestAngleLeft：%f angleSelect = %f",
             closestAngleRight,
             closestAngleLeft, angleSelect);
        return 1;
    } catch (...) {
        LOGE(LOG_TAG, "异常状态12");
        return 0;
    }
}

