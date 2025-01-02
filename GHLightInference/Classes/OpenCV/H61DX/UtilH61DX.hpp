#ifndef UtilH61DX_HPP
#define UtilH61DX_HPP

#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace H61DX {
    /// 红色值
    const int COLOR_RED = 0xFF0000;
    /// 绿色值
    const int COLOR_GREEN = 0x00FF00;
    /// 黄色值
    const int COLOR_YELLOW = 0xFFFF00;
    /// 蓝色值
    const int COLOR_BLUE = 0x0000FF;
    /// 黑色值
    const int COLOR_BLACK = 0x000000;

    /// 日志Tag
    const char * TAG = "Detection-H61DX";

    /// 将BGR颜色转为RGB的int值
    extern inline int BGR2Int(const cv::Vec3b &bgr) {
        return (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
    }

    /// 将BGR转成HSV
    extern inline std::vector<double> BGR2HSV(const cv::Vec3b &bgr) {
        cv::Mat rgb_pixel(1, 1, CV_8UC3, bgr);
        cv::Mat hsv_pixel;
        cv::cvtColor(rgb_pixel, hsv_pixel, cv::COLOR_BGR2HSV);
        
        // 获取转换后的HSV值，并将H值从0-180扩展到0-360
        double h = static_cast<double>(hsv_pixel.at<cv::Vec3b>(0, 0)[0]) * 2; // 扩展H值范围到0-360
        double s = static_cast<double>(hsv_pixel.at<cv::Vec3b>(0, 0)[1]) / 255.0 * 100; // 归一化到0-100%
        double v = static_cast<double>(hsv_pixel.at<cv::Vec3b>(0, 0)[2]) / 255.0 * 100; // 归一化到0-100%
        return {h, s, v};
    }

    /// 将BGR转成LAB
    extern inline std::vector<int> BGR2Lab(const cv::Vec3b &bgr) {
        cv::Mat rgb_pixel(1, 1, CV_8UC3, bgr);
        cv::Mat lab_pixel;
        cv::cvtColor(rgb_pixel, lab_pixel, cv::COLOR_BGR2Lab);
        int l = static_cast<int>(lab_pixel.at<cv::Vec3b>(0, 0)[0]);
        int a = static_cast<int>(lab_pixel.at<cv::Vec3b>(0, 0)[1]) - 128;
        int b = static_cast<int>(lab_pixel.at<cv::Vec3b>(0, 0)[2]) - 128;
        return {l, a, b};
    }
}

#endif // UtilH61DX_HPP
