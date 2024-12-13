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
}

#endif // UtilH61DX_HPP
