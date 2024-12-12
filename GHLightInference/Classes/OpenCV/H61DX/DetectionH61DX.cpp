#include "DetectionH61DX.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "ColorCodingH61DX.hpp"
#include "PictureH61DX.hpp"

using namespace cv;
using namespace std;

DetectionH61DX::DetectionH61DX(int icCount) : icCount(icCount)
{
    _detectionColors = ColorCodingH61DX(icCount).getDetectionColors();
}

DetectionH61DX::~DetectionH61DX()
{
    _originImage.release();
    _nowImage.release();
}

void DetectionH61DX::detection(cv::Mat originImage, std::function<void(std::vector<cv::Point>)> callback)
{
    _originImage = originImage.clone();
    _callback = callback;
}

#if DEBUG

void DetectionH61DX::debugDetection(cv::Mat originImage, std::function<void(std::vector<cv::Mat>)> callback)
{
    cvtColor(originImage, _originImage, COLOR_RGBA2BGR);
    
    // 缩放图片
    Mat resizedImage = PictureH61DX::resizeWithAspectRatio(_originImage, 400);
    
    // 提纯颜色
    Mat replacedImage = PictureH61DX::purifiedImage(resizedImage);
    
    // 开运算2次
    Mat openedImage;
    morphologyEx(replacedImage, openedImage, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 2);
    
    // 闭运算1次
    Mat closeImage;
    morphologyEx(openedImage, closeImage, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 1);
    callback({ resizedImage, replacedImage, openedImage, closeImage });
    
//    _nowImage = PictureH61DX::processImage(_originImage);
//    callback({ _nowImage });
}

#endif
