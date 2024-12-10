#include "DetectionH61DX.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace
{
    const int COLOR_RED = 0xFF0000;
    const int COLOR_GREEN = 0x00FF00;
    const int COLOR_YELLOW = 0xFFFF00;
    const int COLOR_BLUE = 0x0000FF;
    const int COLOR_UNIT_COUNT = 3;
    const int START_IC_COUNT = COLOR_UNIT_COUNT;

    int getBitWidth(int icNum)
    {
        if (icNum <= START_IC_COUNT)
        {
            return 1;
        }
        auto length = icNum - START_IC_COUNT;
        auto bitWidth = 1;
        while (pow(2, bitWidth) * bitWidth * COLOR_UNIT_COUNT < length)
        {
            bitWidth++;
        }
        return bitWidth;
    }

    int getCount(int icNum, int bitWidth)
    {
        if (icNum <= START_IC_COUNT)
        {
            return 0;
        }
        auto length = icNum - 3;
        return length / bitWidth + (length % bitWidth > 0 ? 1 : 0);
    }

    void applyColorToMask(Mat &image, const Mat &hsvImage, const Scalar &lowerBound, const Scalar &upperBound, const Scalar &color)
    {
        Mat mask;
        inRange(hsvImage, lowerBound, upperBound, mask);
        image.setTo(color, mask);
    }
}

DetectionH61DX::DetectionH61DX(int icCount) : icCount(icCount)
{
    _bitWidth = getBitWidth(icCount);
    _count = getCount(icCount, _bitWidth);
    _detectionColors = this->getDetectionColors();
}

DetectionH61DX::~DetectionH61DX()
{
    _originImage.release();
    _nowImage.release();
}

// 获取灯效颜色
std::vector<int> DetectionH61DX::getDetectionColors()
{
    auto result = vector<int>();
    auto start = this->getStartColors();
    result.insert(result.end(), start.begin(), start.end());
    for (int i = 0; i < _count; i++)
    {
        auto colors = this->getIndexColors(i);
        result.insert(result.end(), colors.begin(), colors.end());
    }
    // 将result长度限定在icCount内
    result.resize(icCount);
    return result;
}

// 获取开始颜色
std::vector<int> DetectionH61DX::getStartColors()
{
    return {COLOR_YELLOW, COLOR_GREEN, COLOR_YELLOW};
}

std::vector<int> DetectionH61DX::getIndexColors(int index)
{
    auto colors = vector<int>();
    // 按位遍历index的每一位，如果为0则添加绿色，为1则添加黄色
    for (int i = 0; i < _bitWidth; i++)
    {
        colors.push_back(COLOR_RED);
        if ((index >> i) & 0x01)
        {
            colors.push_back(COLOR_YELLOW);
        }
        else
        {
            colors.push_back(COLOR_GREEN);
        }
        colors.push_back(COLOR_BLUE);
    }
    return colors;
}

void DetectionH61DX::detection(cv::Mat originImage, cv::Mat thresholdImage, std::function<void(std::vector<cv::Point>)> callback)
{
    _originImage = originImage.clone();
    _thresholdImage = thresholdImage.clone();
    _callback = callback;

    // 获取阈值图和原图重叠部分
    cv::bitwise_and(_thresholdImage, _originImage, _nowImage);
    // 提纯图像
    this->purifiedImage();
}

void DetectionH61DX::purifiedImage()
{
    if (_nowImage.empty())
    {
        return;
    }
    Mat hsv_image;
    cvtColor(_nowImage, hsv_image, COLOR_BGR2HSV);
    // 红色
    applyColorToMask(_nowImage, hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), Scalar(255, 0, 0));
    applyColorToMask(_nowImage, hsv_image, Scalar(160, 100, 100), Scalar(180, 255, 255), Scalar(255, 0, 0));
    // 绿色
    applyColorToMask(_nowImage, hsv_image, Scalar(40, 40, 40), Scalar(70, 255, 255), Scalar(0, 255, 0));
    // 蓝色
    applyColorToMask(_nowImage, hsv_image, Scalar(100, 150, 0), Scalar(140, 255, 255), Scalar(0, 0, 255));
    // 黄色
    applyColorToMask(_nowImage, hsv_image, Scalar(20, 100, 100), Scalar(40, 255, 255), Scalar(0, 255, 255));
}

#if DEBUG

void DetectionH61DX::debugDetection(cv::Mat originImage, cv::Mat thresholdImage, std::function<void(std::vector<cv::Mat>)> callback) {
    _originImage = originImage.clone();
    _thresholdImage = thresholdImage.clone();

    // 获取阈值图和原图重叠部分
    cv::bitwise_and(_thresholdImage, _originImage, _nowImage);
    callback({_nowImage});

    // 提纯图像
    this->purifiedImage();
    callback({_nowImage});
}

#endif
