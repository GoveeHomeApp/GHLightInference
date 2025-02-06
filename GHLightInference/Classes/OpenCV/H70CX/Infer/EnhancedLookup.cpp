#include <opencv2/opencv.hpp>
#include <vector>
#include "common.hpp"

using namespace cv;
using namespace std;

class LEDDetector {
private:
    Mat image;
    Mat imageOri;//todo:linpeng
    vector<Point2f> known_positions;
    vector<Point> convex_hull;

    struct LEDFeatures {
        double brightness_threshold;
        int min_area;
        int max_area;
        double circularity_threshold;
        double min_contrast;
    } features;

    void computeConvexHull() {
        vector<Point> points;
        for (const Point2f &p: known_positions) {
            points.push_back(Point(cvRound(p.x), cvRound(p.y)));
        }

        if (points.size() >= 3) {
            convexHull(points, convex_hull);

            // 扩大凸包范围
            Point2f center(0, 0);
            for (const Point &p: convex_hull) {
                center += Point2f(p.x, p.y);
            }
            center *= (1.0f / convex_hull.size());

            vector<Point> expanded_hull;
            float scale = 1.15f;
            for (const Point &p: convex_hull) {
                Point2f vec = Point2f(p.x, p.y) - center;
                Point2f expanded = center + vec * scale;
                expanded_hull.push_back(Point(cvRound(expanded.x), cvRound(expanded.y)));
            }
            convex_hull = expanded_hull;
        }
    }

    bool isPointInConvexHull(const Point2f &point) const {
        if (convex_hull.empty()) return false;
        return pointPolygonTest(convex_hull, point, false) >= 0;
    }

    // 图像增强处理
    Mat enhanceImage(const Mat &input) {
        Mat enhanced;

        // 转换到LAB色彩空间以分离亮度通道
        Mat lab;
        if (input.channels() == 3) {
            cvtColor(input, lab, COLOR_BGR2Lab);
        } else {
            lab = input.clone();
        }

        vector<Mat> lab_planes;
        split(lab, lab_planes);

        // 对L通道进行CLAHE处理
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(5, 5));
        Mat enhanced_l;
        clahe->apply(lab_planes[0], enhanced_l);

        // 增强对比度
        enhanced_l.convertTo(enhanced_l, CV_8UC1, 1.1, 10);

        if (input.channels() == 3) {
            // 合并通道
            lab_planes[0] = enhanced_l;
            merge(lab_planes, lab);
            // 转换回BGR
            cvtColor(lab, enhanced, COLOR_Lab2BGR);
        } else {
            enhanced = enhanced_l;
        }

        return enhanced;
    }

    // 计算点的局部亮度落差
    double calculateLocalContrast(const Point2f& center, const Mat& img) {
        int x = cvRound(center.x);
        int y = cvRound(center.y);
        int radius = 8;

        // 确保ROI不会超出图像边界
        int x1 = max(0, x - radius);
        int y1 = max(0, y - radius);
        int x2 = min(img.cols - 1, x + radius);
        int y2 = min(img.rows - 1, y + radius);

        if (x2 - x1 < radius || y2 - y1 < radius) return 0;  // ROI太小

        // 创建圆形掩码
        Mat mask = Mat::zeros(y2 - y1 + 1, x2 - x1 + 1, CV_8UC1);
        circle(mask, Point(x - x1, y - y1), radius, Scalar(255), -1);

        // 获取ROI区域
        Mat roi = img(Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));

        // 计算圆形区域内的最大和最小亮度
        double minVal, maxVal;
        minMaxLoc(roi, &minVal, &maxVal, nullptr, nullptr, mask);

        return maxVal - minVal;
    }

public:
    LEDDetector(const Mat &img, const vector<Point2f> &known_leds) {
        if (Logger::debugSwitch) {
            imageOri = img.clone();
        }
        // 确保输入图像是灰度图
        if (img.channels() == 3) {
            cvtColor(img, image, COLOR_BGR2GRAY);
        } else {
            image = img.clone();
        }

        known_positions = known_leds;

        features.brightness_threshold = 200;
        features.min_area = 15;
        features.max_area = 200;
        features.circularity_threshold = 0.8;
        features.min_contrast = 30.0;

        computeConvexHull();
    }

    void analyzeKnownLEDs() {
        vector<double> brightnesses;
        vector<double> areas;
        vector<double> circularities;
        vector<double> contrasts;

        for (const Point2f &pos: known_positions) {
            // 计算ROI的范围，确保在图像边界内
            int x = cvRound(pos.x - 8);
            int y = cvRound(pos.y - 8);
            int width = 16;
            int height = 16;

            // 调整ROI的位置和大小，确保不超出图像边界
            if (x < 0) {
                width += x;  // 减少宽度
                x = 0;      // 将x设置为0
            }
            if (y < 0) {
                height += y; // 减少高度
                y = 0;      // 将y设置为0
            }
            if (x + width > image.cols) {
                width = image.cols - x;  // 调整宽度以适应图像边界
            }
            if (y + height > image.rows) {
                height = image.rows - y;  // 调整高度以适应图像边界
            }

            // 检查ROI是否有效
            if (width <= 0 || height <= 0) {
                continue;  // 跳过无效的ROI
            }

            // 创建有效的ROI
            Rect roi(x, y, width, height);
            Mat led_region = image(roi);

            // 计算平均亮度
            Scalar mean_val = mean(led_region);
            brightnesses.push_back(mean_val[0]);

            // 二值化处理
            Mat binary;
            threshold(led_region, binary, features.brightness_threshold, 255, THRESH_BINARY);

            // 确保binary是CV_8UC1格式
            if (binary.type() != CV_8UC1) {
                binary.convertTo(binary, CV_8UC1);
            }

            // 查找轮廓
            vector<vector<Point>> contours;
            findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                double area = contourArea(contours[0]);
                double perimeter = arcLength(contours[0], true);
                if (perimeter > 0) {
                    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
                    areas.push_back(area);
                    circularities.push_back(circularity);
                }
            }

            // 收集亮度落差特征
            double contrast = calculateLocalContrast(pos, image);
            if (contrast > 0) {
                contrasts.push_back(contrast);
            }
        }

        // 更新特征参数
        if (!brightnesses.empty()) {
            double bri = *max_element(brightnesses.begin(), brightnesses.end()) * 1.3;
            features.brightness_threshold =
                    max(bri, 200.0);
            LOGD(LOG_TAG, "更新参数  brightness_threshold=%f  bri=%f",
                 features.brightness_threshold, bri);
        }
        if (!areas.empty()) {
            features.min_area = max(1, cvRound(*min_element(areas.begin(), areas.end()) * 0.8));
            features.max_area = cvRound(*max_element(areas.begin(), areas.end()) * 1.3);
            LOGD(LOG_TAG, "更新参数  min_area=%d  max_area=%d", features.min_area,
                 features.max_area);
        }
        if (!circularities.empty()) {
            features.circularity_threshold = 0.35;  // 使用固定值而不是动态计算
            LOGD(LOG_TAG, "更新参数 circularity_threshold=%f", features.circularity_threshold);
        }
        if (!contrasts.empty()) {
            // 使用较低百分位数作为阈值，避免误删正常点
            sort(contrasts.begin(), contrasts.end());
            int idx = contrasts.size() * 0.2;  // 使用20%分位数
            features.min_contrast = max(20.0, contrasts[idx] * 0.95);  // 设置最小阈值
            LOGD(LOG_TAG, "更新参数 min_contrast=%.1f", features.min_contrast);
        }
    }

    // 添加获取已知点位置的方法，用于访问更新后的位置
    const vector<Point2f> &getKnownPositions() const {
        return known_positions;
    }

    vector<Point2f> findMissingLEDs() {
        vector<Point2f> missing_leds;

        if (convex_hull.empty()) return missing_leds;

        // 创建掩码
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        vector<vector<Point>> hull_contour = {convex_hull};
        fillPoly(mask, hull_contour, Scalar(255));

        // 应用掩码
        Mat masked_image;
        image.copyTo(masked_image, mask);
        Mat enhanced = enhanceImage(masked_image);

        // 预处理
        Mat blurred;
        GaussianBlur(enhanced, blurred, Size(5, 5), 1.5);

        Mat binary;
        threshold(blurred, binary, features.brightness_threshold, 255, THRESH_BINARY);

        // 确保binary是CV_8UC1格式
        if (binary.type() != CV_8UC1) {
            binary.convertTo(binary, CV_8UC1);
        }

        vector<vector<Point>> contours;
        findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 存储需要更新位置的已知点
        vector<pair<int, Point2f>> positions_to_update;

        for (const auto &contour: contours) {
            double area = contourArea(contour);
            if (area < features.min_area || area > features.max_area) continue;

            double perimeter = arcLength(contour, true);
            if (perimeter <= 0) continue;

            double circularity = 4 * CV_PI * area / (perimeter * perimeter);
            if (circularity < features.circularity_threshold) continue;

            Moments m = moments(contour);
            if (m.m00 <= 0) continue;

            Point2f center(m.m10 / m.m00, m.m01 / m.m00);

            if (!isPointInConvexHull(center)) continue;

            // 检查亮度落差
            double contrast = calculateLocalContrast(center, blurred);
            if (contrast < features.min_contrast) {
                LOGD(LOG_TAG, "点位 (%.1f, %.1f) 亮度落差太小: %.1f < %.1f",
                     center.x, center.y, contrast, features.min_contrast);
                continue;
            }

            // 检查与已知点的距离
            bool is_new = true;
            for (size_t i = 0; i < known_positions.size(); i++) {
                float distance = norm(center - known_positions[i]);

                if (distance < 7) {
                    // 距离小于5，视为同一个点，不是新点
                    is_new = false;
                    break;
                } else if (distance <= 10) {
                    // 距离在4-10之间，记录需要更新的位置
                    positions_to_update.push_back({i, center});
                    is_new = false;
                    break;
                }
            }

            if (is_new) {
                missing_leds.push_back(center);
            }
        }

        // 更新需要纠正的位置
        for (const auto &update: positions_to_update) {
            known_positions[update.first] = update.second;
        }

        return missing_leds;
    }

    Mat visualize() {
        Mat result;
        if (!Logger::debugSwitch) {
            return result;
        }
        if (imageOri.channels() == 1) {
            cvtColor(imageOri, result, COLOR_GRAY2BGR);
        } else {
            result = imageOri.clone();
        }

        // 绘制凸包
        if (!convex_hull.empty()) {
            polylines(result, vector<vector<Point>>{convex_hull}, true, Scalar(255, 165, 0), 2);
        }

        // 绘制已知点
        for (const Point2f &pos: known_positions) {
            circle(result, pos, 7, Scalar(0, 255, 0), 1);
        }

        // 绘制新检测点
        vector<Point2f> missing = findMissingLEDs();
        for (const Point2f &pos: missing) {
            circle(result, pos, 7, Scalar(0, 0, 255), 1);

            // 显示亮度落差值
            double contrast = calculateLocalContrast(pos, image);
            putText(result, format("%.1f", contrast),
                    Point(cvRound(pos.x) + 10, cvRound(pos.y)),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
        }

        return result;
    }
};
