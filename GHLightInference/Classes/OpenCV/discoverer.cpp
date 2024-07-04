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
findByContours(Mat &image, vector<Point> &pointVector, vector<LightPoint> &lightPoints, int icNum,
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
//        vLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * vFactor);
    }

    // 应用LUT到S通道
    cv::LUT(hsvChannels[1], sLUT, hsvChannels[1]);
//    // 应用LUT到V通道
//    cv::LUT(hsvChannels[2], vLUT, hsvChannels[2]);
////
////    // 合并增强后的通道回HSV图像
    cv::merge(hsvChannels, hsv);
////    // 将HSV图像转换回BGR颜色空间以便显示
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
    std::vector<std::vector<Point>> contoursA;
    std::vector<std::vector<Point>> contoursB;
    findContours(dst, contoursA, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(dst2, contoursB, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contoursA.size(); i++) {
        vector<Point> contour = contoursA[i];
        double contourArea = cv::contourArea(contour);
        if (contourArea < 1.0)continue;
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        pointVector.push_back(Point2i(center.x, center.y));
        LightPoint lp = LightPoint();
        lp.point2f = center;
        lightPoints.push_back(lp);
        LOGV("contourArea", "drawColorMask contourArea: %f  radius: %f", contourArea, radius);
    }
    for (int i = 0; i < contoursB.size(); i++) {
        vector<Point> contour = contoursB[i];
        double contourArea = cv::contourArea(contour);
        if (contourArea < 1.0)continue;
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        center.x = static_cast<int>(center.x);
        center.y = static_cast<int>(center.y);
        pointVector.push_back(Point2i(center.x, center.y));
        LightPoint lp = LightPoint();
        lp.point2f = center;
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

void findNoodleLamp(Mat &image, vector<Point> &pointVector, vector<LightPoint> &lightPoints,
                    vector<Mat> &outMats) {
    LOGD(LOG_TAG, "=====>查找灯带光点");
    // 将图像从BGR转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 分离HSV通道
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    // 定义增强系数
    double sFactor = 1.2; // 饱和度增强系数
    // 创建S和V通道的LUT
    int lutSize = 256;
    cv::Mat sLUT(1, lutSize, CV_8UC1);
    cv::Mat vLUT(1, lutSize, CV_8UC1);
    for (int i = 0; i < lutSize; ++i) {
        // 增强饱和度
        sLUT.at<uchar>(i) = cv::saturate_cast<uchar>(i * sFactor);
        // 增强亮度
    }

    // 应用LUT到S通道
    cv::LUT(hsvChannels[1], sLUT, hsvChannels[1]);
    // 合并增强后的通道回HSV图像
    cv::merge(hsvChannels, hsv);
    // 将HSV图像转换回BGR颜色空间以便显示
    cv::Mat enhanced, enhancedGay;
    cv::cvtColor(hsv, enhanced, cv::COLOR_HSV2BGR);
    outMats.push_back(enhanced);

    Mat gray;
    cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);

    lightPoints.clear();
    Mat dst = thresholdNoodleLamp(gray, pointVector, lightPoints, outMats);
}

vector<int>
polyPoints(vector<Point2i> &pointVector, int k, double stddevThreshold, Mat &outMat) {
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
double distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算线段的角度
double angle(Vec4i line) {
    return atan2(line[3] - line[1], line[2] - line[0]);
}

// 计算线段长度
double lineLength(const Vec4i &line) {
    return distance(Point(line[0], line[1]), Point(line[2], line[3]));
}

double distanceBetweenPoints(Point2f p1, Point2f p2) {
    Point2f diff = p1 - p2;
    return sqrt(diff.x * diff.x + diff.y * diff.y);
}

vector<vector<ContourInfo>>
findAlignedContours(vector<vector<Point>> &contours, int lineMinDistance = 5,
                    int minGroupCnt = 6,
                    int maxGroupCnt = 15) {
    vector<ContourInfo> validContours;
    for (const auto &contour: contours) {
        double area = contourArea(contour);
        if (area >= 20 && area <= 200) {
            Moments m = moments(contour);
            Point2f center(m.m10 / m.m00, m.m01 / m.m00);
            validContours.push_back({contour, center, area, minAreaRect(contour)});
        }
    }
//    LOGD(LOG_TAG, "有效矩形数量 = %d，全部数量 = %d", validContours.size(), contours.size());
    vector<vector<ContourInfo>> lineGroups;
    vector<bool> used(validContours.size(), false);

    for (size_t i = 0; i < validContours.size(); ++i) {
        if (used[i]) continue;

        vector<ContourInfo> currentGroup = {validContours[i]};
        used[i] = true;

        for (size_t j = 0; j < validContours.size(); ++j) {
            if (i == j || used[j]) continue;

            vector<ContourInfo> testGroup = currentGroup;
            testGroup.push_back(validContours[j]);

            // 检查是否所有轮廓都大致在一条直线上
            Vec4f line;
            vector<Point2i> centers;
            for (const auto &c: testGroup) {
                centers.push_back(c.center);
            }

            fitLine(centers, line, DIST_L2, 0, 0.01, 0.01);

            bool allAligned = true;
            double maxDistance = 0;
            for (const auto &c: testGroup) {
                Point2f vec(line[0], line[1]);
                Point2f ptOnLine(line[2], line[3]);
                Point2f ptToCenter = c.center - ptOnLine;
                float distance = abs(ptToCenter.cross(vec));
                if (distance > lineMinDistance) {  // 允许的最大偏差
                    allAligned = false;
                    break;
                }
                maxDistance = max(maxDistance,
                                  distanceBetweenPoints(c.center, testGroup[0].center));
            }
//            LOGV(LOG_TAG, "检查间距 maxDistance= %f size = %d", maxDistance, testGroup.size());
            // 检查间距
            if (allAligned && maxDistance <= 35 * (testGroup.size() - 1)) {
                currentGroup = testGroup;
                if (testGroup.size() <= 11 || testGroup.size() >= 6) {
                    used[j] = true;
                }
            }
        }
//        LOGV(LOG_TAG, "currentGroup = %d", currentGroup.size());
        if (currentGroup.size() >= minGroupCnt && currentGroup.size() <= maxGroupCnt) {
            lineGroups.push_back(currentGroup);
        }
    }

    LOGD(LOG_TAG, "lineGroups = %d", lineGroups.size());
    return lineGroups;
}

bool hasSignificantOverlap(const std::vector<cv::RotatedRect> &rects, double overlapThreshold) {
    for (size_t i = 0; i < rects.size(); ++i) {
        for (size_t j = i + 1; j < rects.size(); ++j) {
            // 计算两个旋转矩形的交集
            cv::Rect intersectRect = rects[i].boundingRect() & rects[j].boundingRect();
            double intersectArea = intersectRect.area();
            double totalArea =
                    rects[i].boundingRect().area() + rects[j].boundingRect().area() - intersectArea;
            if (intersectArea / totalArea > overlapThreshold) {
                return true; // 发现重叠面积超过阈值
            }
        }
    }
    return false; // 没有发现重叠面积超过阈值
}

void polyContours(vector<Point2i> &pointVector, vector<ContourInfo> &groups, int k,
                  double stddevThreshold) {
    vector<int> eraseVector;
    if (pointVector.empty()) {
        LOGE(LOG_TAG, "polyPoints null");
        return;
    }
    if (pointVector.size() < k + 4)
        return;
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
                if (pointVector.begin() + i < pointVector.end()) {
                    pointVector.erase(pointVector.begin() + i);
                    groups.erase(groups.begin() + i);
                    eraseVector.push_back(i);
                }
            }
        }
        LOGD(LOG_TAG, "pointVector擦除离群点 = %d", size - pointVector.size());
    } catch (...) {
        LOGE(LOG_TAG, "========》 异常7");
    }
}

Mat thresholdNoodleLamp(Mat &src, vector<Point> &pointVector, vector<LightPoint> &lightPoints,
                        vector<Mat> &outMats) {
    lightPoints.clear();
    Mat morphology_image1, morphology_image, threshold_image, gray = src.clone();
    int thresh = 205;
    threshold(src, threshold_image, thresh, 255, THRESH_BINARY);

    morphology_image1 = morphologyImage(threshold_image, 3, 5, MORPH_ELLIPSE);

    outMats.push_back(morphology_image1);

    vector<vector<Point>> contoursPoint;
    findContours(morphology_image1, contoursPoint, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<ContourInfo>> alignedGroups = findAlignedContours(contoursPoint);

//    LOGE(LOG_TAG, "-------alignedGroups = %d", alignedGroups.size());
//    LOGE(LOG_TAG, "-------lightPoints = %d", lightPoints.size());

    Mat result = src.clone();

    for (size_t i = 0; i < alignedGroups.size(); i++) {
        vector<ContourInfo> group = alignedGroups[i];
        vector<Point2f> allPoints;
        vector<Point2i> cPoints;
        for (const auto &contourInfo: group) {
            cPoints.push_back(contourInfo.center);
        }
        polyContours(cPoints, group, 1, 1.2);

        for (const auto &contourInfo: group) {
            vector<Point2i> contourPoints = contourInfo.contour;

            RotatedRect minRotatedRect = minAreaRect(contourPoints);

            drawContours(result, vector<vector<Point>>{contourPoints}, -1, Scalar(0, 255, 0),
                         2);
            Point2f vertices[4];
            minRotatedRect.points(vertices);
            for (int j = 0; j < 4; j++) {
                allPoints.push_back(vertices[j]);
            }
        }
        RotatedRect groupRect = minAreaRect(allPoints);
        Point2f groupVertices[4];
        groupRect.points(groupVertices);
        LightPoint lp = LightPoint();
        lp.with = groupRect.size.width;
        lp.height = groupRect.size.height;
        lp.tfRect = groupRect.boundingRect();
        lp.rotatedRect = groupRect;
        lp.point2f = groupRect.center;
        LOGD(LOG_TAG, "矩形LightPoint  %f x %f  center=(%f,%f)", groupRect.size.width,
             groupRect.size.height, groupRect.center.x, groupRect.center.y);
        pointVector.push_back(groupRect.center);
        lightPoints.push_back(lp);
        for (int k = 0; k < 4; k++) {
            line(result, groupVertices[k], groupVertices[(k + 1) % 4], Scalar(0, 0, 255), 2);
        }
    }
    outMats.push_back(result);
    LOGW(LOG_TAG, "矩形轮廓数量lightPoints= %d", lightPoints.size());
    return morphology_image1;
}

Mat thresholdPoints(Mat &src, Mat &bgrSrc, Mat &hue, int color,
                    vector<Mat> &outMats) {
    Mat morphology_image, threshold_image;
    int contoursSize = 1200;
    int thresh = 200;
    // 寻找轮廓
    std::vector<std::vector<Point>> contours;
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
//                erode(dst, dst, kernelErode, Point(-1, -1), 1);
                erode(dst, dst, kernelErodeMin);
                outMats.push_back(dst);
            }
            cv::drawContours(dst, contours, static_cast<int>(i), cv::Scalar::all(255), -1);
        }
    }
    outMats.push_back(dst);
    return dst;
}

double calculateDistance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void mergePoints(vector<Point> &points, double threshold) {
    try {
        vector<Point> mergedPoints;
        vector<bool> merged(points.size(), false);

        for (int i = 0; i < points.size(); i++) {
            if (merged[i]) continue;

            Point mergedPoint = points[i];
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
        morphologyEx(outMat, outMat, MORPH_DILATE, dilateKernel, Point(-1, -1),
                     1);
    }
    return outMat;
}

// 计算两点之间的距离
double distanceP(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算包含所有点的最小等腰梯形
int getMinTrapezoid(Mat &image, const vector<Point> &pointsSrc, vector<Point> &trapezoid4Points) {
    if (pointsSrc.empty() || image.empty()) {
        LOGE(LOG_TAG, "getMinTrapezoid null");
        return 0;
    }
    try {
        vector<Point> points(pointsSrc);
        polyPoints(points,
                   3, 1.9, image);
        vector<Point2i> hull;
        convexHull(points, hull);
        vector<double> angleVector;
        vector<bool> rightVector;
        //左右凸包最最靠边角点
        Point2i pointRight(0, 0), pointLeft(0, 0);
        // 计算凸包的中心点
        Moments mu = moments(hull);
        for (int i = 0; i < hull.size(); i++) {
            Point2i point1 = hull[i];
            Point2i point2 = hull[(i + 1) % hull.size()];
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
        line(image, Point2i(mu.m10 / mu.m00, 0), Point2i(mu.m10 / mu.m00, image.rows),
             Scalar(255, 255, 255), 2);

        double averageSlope = 76;
        LOGD(LOG_TAG, "均值斜率：%f", averageSlope);
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
            pointRight.x = pointRight.x;
            //取右边点
            int leftX = mu.m10 / mu.m00 - (pointRight.x - mu.m10 / mu.m00);
            pointLeft = Point(leftX, pointRight.y);
        } else {
            pointLeft.x = pointLeft.x;
            int rightX = mu.m10 / mu.m00 + (mu.m10 / mu.m00 - pointLeft.x);
            pointRight = Point(rightX, pointLeft.y);
        }
        if (abs(closestAngleRight) > abs(closestAngleLeft)) {
            angleSelect = abs(closestAngleRight);
        } else {
            angleSelect = abs(closestAngleLeft);
        }
        if (angleSelect <= 5) {
            LOGE(LOG_TAG, "左右均无有效斜边");
            return 0;
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

        line(image, pointLeft, Point2i(leftTopX, minY),
             Scalar(0, 0, 255),
             3);

        line(image, pointRight, Point2i(rightTopX, minY),
             Scalar(0, 0, 255), 3);
        trapezoid4Points.push_back(Point2i(rightTopX, minY));
        trapezoid4Points.push_back(pointRight);
        trapezoid4Points.push_back(pointLeft);
        trapezoid4Points.push_back(Point2i(leftTopX, minY));

        LOGD(LOG_TAG, "closestAngleRight：%f   closestAngleLeft：%f angleSelect = %f",
             closestAngleRight,
             closestAngleLeft, angleSelect);
        return 1;
    } catch (...) {
        LOGE(LOG_TAG, "异常状态12");
        return 0;
    }
}

