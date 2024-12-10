#ifndef DetectionH61DX_HPP
#define DetectionH61DX_HPP

#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>

class DetectionH61DX
{
public:
    /// @brief ic个数
    int icCount;

    /// @brief 构造函数
    /// @param icCount ic个数
    DetectionH61DX(int icCount = 70);

    ~DetectionH61DX();

    /// @brief 灯效展示颜色序列
    std::vector<int> getDetectionColors();

    /// @brief 获取起始的颜色序列
    std::vector<int> getStartColors();

    /// @brief 识别图像
    /// @param originImage 原始图像
    /// @param thresholdImage 阈值图像
    /// @param callback 识别结果回调函数
    void detection(cv::Mat originImage, cv::Mat thresholdImage, std::function<void(std::vector<cv::Point>)> callback);
    
#if DEBUG
    void debugDetection(cv::Mat originImage, cv::Mat thresholdImage, std::function<void(std::vector<cv::Mat>)> callback);
#endif

private:
    /// @brief 识别的颜色编码
    std::vector<int> _detectionColors;
    /// @brief 编码宽度
    int _bitWidth;
    /// @brief 编码个数
    int _count;
    /// @brief 原始图像
    cv::Mat _originImage;
    /// @brief 阈值图像
    cv::Mat _thresholdImage;
    /// @brief 处理后的图像
    cv::Mat _nowImage;
    /// @brief 识别结果回调函数
    std::function<void(std::vector<cv::Point>)> _callback;

    /// @brief 获取对应序号的颜色编码
    /// @param index 编码序号
    /// @return 对应的颜色编码
    std::vector<int> getIndexColors(int index);

    /// @brief 提纯图像
    void purifiedImage();
};

#endif // DetectionH61DX_HPP
