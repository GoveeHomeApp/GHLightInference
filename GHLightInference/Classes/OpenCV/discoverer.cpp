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
findByContours(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints,
               vector<Mat> &outMats) {
    LOGD(LOG_TAG, "===============查找灯带光点===============");
    // 1. HSV转换和饱和度增强
    Mat hsv, enhanced;
    vector<Mat> hsvChannels;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    split(hsv, hsvChannels);

    // 创建并应用饱和度LUT
    const int lutSize = 256;
    const double sFactor = 1.2;
    Mat sLUT(1, lutSize, CV_8UC1);
    for (int i = 0; i < lutSize; ++i) {
        sLUT.at<uchar>(i) = saturate_cast<uchar>(i * sFactor);
    }
    LUT(hsvChannels[1], sLUT, hsvChannels[1]);

    // 合并通道并转回BGR
    merge(hsvChannels, hsv);
    cvtColor(hsv, enhanced, COLOR_HSV2BGR);
    cvtColor(enhanced, enhanced, COLOR_BGR2GRAY);
    outMats.push_back(enhanced);

    // 3. 阈值处理和轮廓检测
    Mat binaryImage = thresholdPoints(enhanced, hsvChannels[0], outMats);

    // 确保二值图像类型正确
    if (binaryImage.type() != CV_8UC1) {
        Mat temp;
        binaryImage.convertTo(temp, CV_8UC1);
        binaryImage = temp;
    }
    outMats.push_back(binaryImage);

    // 4. 轮廓检测
    vector<vector<Point>> contours;
    findContours(binaryImage.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    LOGD(LOG_TAG, "=====contours count =%d", contours.size());

    // 5. 处理轮廓
    for (const auto &contour: contours) {
        double area = contourArea(contour);
        if (area < 10.0) continue;

        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
//        if (radius < 3)continue;
        pointVector.emplace_back(center);

        LightPoint lightPoint;
        lightPoint.position = center;
        lightPoints.push_back(lightPoint);
//        LOGV("contourArea", "Light point area: %.2f, radius: %.2f", area, radius);
    }
    LOGD(LOG_TAG, "点数量=%d", pointVector.size());
}

Mat thresholdPoints(Mat &src, Mat &hue,
                    vector<Mat> &outMats) {
    Mat morphology_image, threshold_image;
    int thresh = 200;
    // 1. 自适应阈值查找
    vector<vector<Point>> contours;
    threshold(src, threshold_image, thresh, 255, THRESH_BINARY);
    morphology_image = morphologyImage(threshold_image, 3, 5, MORPH_ELLIPSE);
    findContours(morphology_image.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    outMats.push_back(morphology_image);

    // 2. 轮廓排序
    sort(contours.begin(), contours.end(), compareContourAreas);

    // 3. 轮廓处理
    Mat dst = Mat::zeros(morphology_image.size(), CV_8UC1);
    Mat kernelErode = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    bool hasErode = false;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // 创建掩码
        Mat mask = Mat::zeros(hue.size(), CV_8UC1);
        Point2f center;
        float radius = 9.0f;
        minEnclosingCircle(contours[i], center, radius);
        circle(mask, center, 9, Scalar(255), -1);
        drawContours(mask, contours, i, Scalar(255), FILLED);

        // 计算统计值
        Scalar mean, stddev;
        meanStdDev(hue, mean, stddev, mask);
//        LOGV(LOG_TAG, "Area: %.2f, Mean: %.2f, StdDev: %.2f", area, mean[0], stddev[0]);

        // 根据面积和标准差决定处理方式
        if (area > 400 || stddev[0] < 4) {
//            LOGV(LOG_TAG, "Skipped: Area=%.2f, StdDev=%.2f", area, stddev[0]);
            continue;
        }

        if (area >= 200) {
            drawContours(dst, contours, i, Scalar(255), -1);
        } else {
            if (!hasErode) {
                erode(dst, dst, kernelErode);
                outMats.push_back(dst);
                hasErode = true;
            }
            if (area > 20)
                drawContours(dst, contours, i, Scalar(255), -1);
        }
    }

    outMats.push_back(dst);
    return dst;
}

void findNoodleLamp(Mat &image, vector<Point2f> &pointVector, vector<LightPoint> &lightPoints,
                    vector<Mat> &outMats) {
    LOGD(LOG_TAG, "tf识别数量：%d", lightPoints.size());
    Mat tfOut = image.clone();
    for (auto lp: lightPoints) {
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
polyPoints(vector<Point2f> &pointVector, int k, double stddevThreshold) {
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
        vector<double> distances;
        map<int, vector<float>> distancesMap;
        for (int i = 0; i < pointsMat.rows; i++) {
            Point2f point = pointsMat.at<Point2f>(i);
            int type = labels.at<int>(i);
            Point2f center = centers.at<Point2f>(type);
            double distance = norm(point - center);
            distances.push_back(distance);
        }
        // 计算离群点的阈值
        Scalar mean, stddev;
        meanStdDev(distances, mean, stddev);
        double threshold = mean[0] + stddevThreshold * stddev[0];
        int size = pointVector.size();

        // 输出离群点
        for (int i = pointsMat.rows - 1; i >= 0; i--) {
            if (distances[i] > threshold) {
//                circle(outMat, pointVector[i], 10, Scalar(0, 0, 0), 2);
                if (pointVector.begin() + i < pointVector.end()) {
                    pointVector.erase(pointVector.begin() + i);
                    eraseVector.push_back(i);
                }
            }
        }

        LOGD(LOG_TAG, "pointVector擦除离群点 = %d", size - pointVector.size());
        vector<vec4f> data;
        for (auto &i: pointVector) {
            data.push_back(vec4f{i.x * 1.f, i.y * 1.f});
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

Rect2i safeRect(const Rect2i &region, const Size &imageSize) {
    Rect2i safe = region;
    safe.x = safe.x - 4;
    safe.y = safe.y - 4;
    safe.width = safe.width + 4;
    safe.height = safe.height + 4;
    safe.x = max(0, min(safe.x, imageSize.width - 1));
    safe.y = max(0, min(safe.y, imageSize.height - 1));
    safe.width = min(safe.width, imageSize.width - safe.x);
    safe.height = min(safe.height, imageSize.height - safe.y);
    return safe;
}


RotatedRect findLightStripInRegion(const Mat &input, const Rect2i &region, vector<Mat> &outMats) {
    Rect2i safeRegion = safeRect(region, input.size());
    if (safeRegion.width <= 0 || safeRegion.height <= 0) {
        return {}; // 返回一个空的RotatedRect
    }
    Mat gray, hsv, binary2;
    Mat roi = input(safeRegion);
    // 转换为灰度图
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    Mat result = roi.clone();

    cvtColor(roi, hsv, COLOR_BGR2HSV);
    // 提取亮度通道
    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);
    Mat value = hsvChannels[2];

    // 应用高斯模糊以减少噪声
    GaussianBlur(value, value, Size(5, 5), 0);

    // 定义绿色和红色灯条的颜色范围（根据实际情况调整）
    Scalar lower_green(35, 50, 50);
    Scalar upper_green(85, 255, 255);

    Scalar lower_red1(0, 50, 50);
    Scalar upper_red1(10, 255, 255);
    Scalar lower_red2(170, 50, 50);
    Scalar upper_red2(180, 255, 255);

    // 颜色分割
    Mat mask_green, mask_red1, mask_red2, mask_red, mask;
    inRange(hsv, lower_green, upper_green, mask_green);
    inRange(hsv, lower_red1, upper_red1, mask_red1);
    inRange(hsv, lower_red2, upper_red2, mask_red2);

    // 合并红色和绿色的掩码
    mask_red = mask_red1 | mask_red2;
    mask = mask_green | mask_red;

    // 使用Otsu方法进行阈值处理
    double otsuThresh = threshold(value, binary2, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Mat binary1;
    Mat dst = Mat::zeros(roi.size(), CV_8UC1);
    int maxIterations = 8;
    otsuThresh = otsuThresh + 10;
    double minContourArea = 200;
    Mat resultGray = gray.clone();
    int stopIterations = 0;
    for (int i = 0; i < maxIterations && otsuThresh < 235; ++i) {
        stopIterations = i;
        // 应用二值化
        threshold(gray, binary1, otsuThresh, 255, THRESH_BINARY);

        // 查找轮廓
        vector<vector<Point>> contours;
        findContours(binary1, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 对大面积区域应用更高的阈值
        Mat roiIteration;
        // 遍历大面积轮廓
        for (const auto &contour: contours) {
            double area = contourArea(contour);
            if (area > minContourArea) {
                // 创建掩码
                Mat maskIteration = Mat::zeros(roi.size(), CV_8UC1);
                drawContours(maskIteration, vector<vector<Point>>{contour}, 0,
                             Scalar(255), -1);

                resultGray.copyTo(roiIteration, maskIteration);
            } else {
                if (i < 3 && area < 30) {
                } else {
                    drawContours(dst, vector<vector<Point>>{contour}, 0,
                                 Scalar(255), -1);
                }
            }
        }
        resultGray = roiIteration;
        // 增加阈值，为下一次迭代做准备
        otsuThresh += 6;
    }
//    outMats.push_back(dst);

    mask = mask & dst;

    LOGD(LOG_TAG, "otsuThresh = %f  stopIterations = %d", otsuThresh, stopIterations);
    // 形态学操作
    Mat morphed1;
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(mask, morphed1, MORPH_CLOSE, kernel1);
    morphologyEx(morphed1, morphed1, MORPH_OPEN, kernel2);
//    outMats.push_back(morphed1);

    // 查找轮廓
    vector<vector<Point>> contours1;
    findContours(morphed1, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


    // 合并所有轮廓的点集
    vector<Point> allPoints;
    for (const auto &contour: contours1) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    if (allPoints.empty()) {
        LOGE(LOG_TAG, "allPoints empty !");
        return {};
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
        for (auto resultRect: rRectList) {
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
    morphologyEx(image, outMat, MORPH_OPEN, openKernel);// 开运算去除小噪点
    morphologyEx(image, outMat, MORPH_CLOSE, openKernel);// 闭运算填充内部小孔
    if (dilateKernelSize > 0) {//腐蚀操作，减小轮廓大小到实际尺寸
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
        polyPoints(points, 3, 1.9);
        vector<Point2f> hull;
        convexHull(points, hull);
        vector<double> angleVector;
        vector<bool> rightVector;
        //左右凸包最最靠边角点
        Point2f pointRight(0, 0), pointLeft(0, 0);
        // 计算凸包的中心点
        Moments mu = moments(hull);
        for (const auto &item: points){
            if (item.x <= 10 || item.x >= image.cols - 10 || item.y <= 0|| item.y >= image.rows - 10) {
                LOGE(LOG_TAG, "getMinTrapezoid 异常点 = %f x %f", item.x, item.y);
            }
            circle(image, item, 7, Scalar(0, 255, 255, 150), 2);
        }
        // 绘制凸包
//        if (!hull.empty()) {
//            polylines(image, vector<vector<Point2f>>{hull}, true, Scalar(255, 165, 0), 2);
//        }

        for (const auto &item: hull) {
            LOGD(LOG_TAG, "item = %f - %f", item.x, item.y);
        }

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
    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "getMinTrapezoid e =  %s", e.what());
        return 0;
    } catch (...) {
        LOGE(LOG_TAG, "异常状态12");
        return 0;
    }
}

