#include <opencv2/opencv.hpp>
#include <vector>
#include "common.hpp"

using namespace cv;
using namespace std;

class LEDDetector {
private:
    Mat image;
    vector<Point2f> known_positions;
    vector<Point> convex_hull;

    struct LEDFeatures {
        double brightness_threshold;
        int min_area;
        int max_area;
        double circularity_threshold;
    } features;

    void computeConvexHull() {
        vector<Point> points;
        for (const Point2f &p: known_positions) {
            points.push_back(Point(cvRound(p.x), cvRound(p.y)));
        }

        if (points.size() >= 3) {  // 确保有足够的点来形成凸包
            convexHull(points, convex_hull);

            // 扩大凸包范围
            Point2f center(0, 0);
            for (const Point &p: convex_hull) {
                center += Point2f(p.x, p.y);
            }
            center *= (1.0f / convex_hull.size());

            vector<Point> expanded_hull;
            float scale = 1.05f;
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

public:
    LEDDetector(const Mat &img, const vector<Point2f> &known_leds) {
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

        computeConvexHull();
    }

    void analyzeKnownLEDs() {
        vector<double> brightnesses;
        vector<double> areas;
        vector<double> circularities;

        for (const Point2f &pos: known_positions) {
            // 确保ROI在图像范围内
            Rect roi(
                    max(0, cvRound(pos.x - 5)),
                    max(0, cvRound(pos.y - 5)),
                    min(image.cols - cvRound(pos.x - 10), 10),
                    min(image.rows - cvRound(pos.y - 10), 10)
            );
            if (roi.width <= 0 || roi.height <= 0) continue;

            Mat led_region = image(roi);

            Scalar mean_val = mean(led_region);
            brightnesses.push_back(mean_val[0]);

            Mat binary;
            threshold(led_region, binary, features.brightness_threshold, 255, THRESH_BINARY);

            // 确保binary是CV_8UC1格式
            if (binary.type() != CV_8UC1) {
                binary.convertTo(binary, CV_8UC1);
            }

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
        }

        // 更新参数
        if (!brightnesses.empty()) {
            features.brightness_threshold =
                    *max_element(brightnesses.begin(), brightnesses.end()) * 0.85;
            LOGD(LOG_TAG, "更新参数  brightness_threshold=%f", features.brightness_threshold);
//            features.brightness_threshold = 200;
            features.max_area = min(features.max_area, 200);
        }
        if (!areas.empty()) {
            features.min_area = max(1, cvRound(*min_element(areas.begin(), areas.end()) * 0.8));
            features.max_area = cvRound(*max_element(areas.begin(), areas.end()) * 1.2);
            LOGD(LOG_TAG, "更新参数  min_area=%d  max_area=%d", features.min_area,
                 features.max_area);
        }
        if (!circularities.empty()) {
            features.circularity_threshold =
                    *min_element(circularities.begin(), circularities.end()) * 0.9;
            LOGD(LOG_TAG, "更新参数 circularity_threshold=%f", features.circularity_threshold);
            features.circularity_threshold = 0.3;
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

        // 预处理
        Mat blurred;
        GaussianBlur(masked_image, blurred, Size(3, 3), 1.5);

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

            // 检查与已知点的距离
            bool is_new = true;
            for (size_t i = 0; i < known_positions.size(); i++) {
                float distance = norm(center - known_positions[i]);

                if (distance < 5) {
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
        if (image.channels() == 1) {
            cvtColor(image, result, COLOR_GRAY2BGR);
        } else {
            result = image.clone();
        }

        // 绘制凸包
        if (!convex_hull.empty()) {
            polylines(result, vector<vector<Point>>{convex_hull}, true, Scalar(255, 165, 0), 2);
        }

        // 绘制已知点
        for (const Point2f &pos: known_positions) {
            circle(result, pos, 5, Scalar(0, 255, 0), -1);
            circle(result, pos, 7, Scalar(0, 255, 0), 1);
        }

        // 绘制新检测点
        vector<Point2f> missing = findMissingLEDs();
        for (const Point2f &pos: missing) {
            circle(result, pos, 5, Scalar(0, 0, 255), -1);
            circle(result, pos, 7, Scalar(0, 0, 255), 1);
        }

        return result;
    }
};

// 使用示例
//int main() {
//    // 读取图像
//    Mat image = imread("led_image.jpg", IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        cout << "Error: Could not read the image." << endl;
//        return -1;
//    }
//
//    // 已知灯珠位置
//    vector<Point2f> known_positions = {
//            Point2f(100, 100),
//            Point2f(200, 200),
//            Point2f(300, 300)
//    };
//
//    LEDDetector detector(image, known_positions);
//    detector.analyzeKnownLEDs();
//
//    Mat result = detector.visualize();
//    imshow("LED Detection Result", result);
//    waitKey(0);
//
//    return 0;
//}
