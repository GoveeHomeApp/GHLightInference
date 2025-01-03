#ifndef PictureH61DX_HPP
#define PictureH61DX_HPP

#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

/// 图片处理
class PictureH61DX
{
public:
    /// 处理图片
    static cv::Mat processImage(cv::Mat& image);

    #if DEBUG
    static cv::Mat debugProcessImage(cv::Mat& image, std::function<void(cv::Mat&)> callback);
    #endif
    
private:
    /// @brief 提纯图像，用纯的红绿蓝黄替换对应颜色，否则则为黑色
    static cv::Mat purifiedImage(cv::Mat& image);
    
    /// 缩放图片到目标大小（以短边为基准）
    static cv::Mat resizeWithAspectRatio(const cv::Mat& image, int target, int inter = cv::INTER_AREA);

    /// 过滤掉低亮度的像素和毛刺
    static cv::Mat filterLowBrightnessAndburrs(cv::Mat& image, int threshold = 100);

};

#endif // PictureH61DX_HPP
