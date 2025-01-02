#ifndef DetectionH61DX_HPP
#define DetectionH61DX_HPP

#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>

/// H61DX灯带识别
class DetectionH61DX
{
public:
    /// @brief ic个数
    int icCount;

    /// @brief 构造函数
    /// @param icCount ic个数
    DetectionH61DX(int icCount = 70);

    ~DetectionH61DX();
    
    /// @brief 识别图像
    /// @param originImage 原始图像
    /// @param callback 识别结果回调函数
    void detection(cv::Mat originImage, std::function<void(std::vector<cv::Point>)> callback);
    
#if DEBUG
    std::vector<cv::Point> debugDetection(cv::Mat originImage, std::function<void(std::vector<cv::Mat>)> callback);
#endif

private:
    /// @brief 识别的颜色编码
    std::vector<int> _detectionColors;
    /// @brief 原始图像
    cv::Mat _originImage;
    /// @brief 处理后的图像
    cv::Mat _nowImage;
    /// @brief 识别结果回调函数
    std::function<void(std::vector<cv::Point>)> _callback;
};

#endif // DetectionH61DX_HPP
