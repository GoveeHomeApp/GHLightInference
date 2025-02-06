#include "PictureH61DX.hpp"
#include "GroupUtilH61DX.hpp"
#include "UtilH61DX.hpp"

using namespace cv;
using namespace std;
using namespace H61DX;

namespace
{
    // 图片缩放目标宽度
    const int TARGET_WIDTH = 400;
}

cv::Mat PictureH61DX::processImage(cv::Mat &image)
{
    if (image.empty())
    {
        return image.clone();
    }
    // 缩放图片
    Mat resizedImage = resizeWithAspectRatio(image, TARGET_WIDTH);
    // 去除低亮度像素和毛刺
    Mat brightnessImage = filterLowBrightnessAndburrs(resizedImage);
    // 提纯颜色
    Mat replacedImage = purifiedImage(brightnessImage);
    return replacedImage;
}

#if DEBUG
cv::Mat PictureH61DX::debugProcessImage(cv::Mat &image, std::function<void(cv::Mat &)> callback)
{
    if (image.empty())
    {
        return image.clone();
    }
    // 缩放图片
    Mat resizedImage = resizeWithAspectRatio(image, TARGET_WIDTH);
    callback(resizedImage);

    // 低亮度过滤
    Mat hsv_image;
    cvtColor(resizedImage, hsv_image, COLOR_BGR2HSV);
    auto lower = Scalar(0, 0, 100);
    auto upper = Scalar(180, 255, 255);
    Mat mask;
    inRange(hsv_image, lower, upper, mask);

    // 对原图进行掩膜操作
    Mat brightnessImage;
    bitwise_and(resizedImage, resizedImage, brightnessImage, mask);
    callback(brightnessImage);

    auto span = max(3, GroupUtilH61DX::getSpan(brightnessImage));
    if (span > 3)
    {
        // 去除毛刺
        morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
        morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
        Mat targetImage;
        bitwise_and(resizedImage, resizedImage, targetImage, mask);
        brightnessImage = targetImage;
        callback(targetImage);
    }

    // 提纯颜色
    Mat replacedImage = purifiedImage(brightnessImage);
    callback(replacedImage);

    return replacedImage;
}
#endif

cv::Mat PictureH61DX::resizeWithAspectRatio(const cv::Mat &image, int target, int inter)
{
    Size dim;
    int h = image.rows;
    int w = image.cols;

    if (w < h)
    {
        float r = static_cast<float>(target) / w;
        dim = Size(target, static_cast<int>(h * r));
    }
    else
    {
        float r = static_cast<float>(target) / h;
        dim = Size(static_cast<int>(w * r), target);
    }

    Mat resized;
    resize(image, resized, dim, 0, 0, inter);
    return resized;
}

cv::Mat PictureH61DX::purifiedImage(cv::Mat &image)
{
    auto resultImage = image.clone();
    if (image.empty())
    {
        return resultImage;
    }
    int threshold = 100;
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            Vec3b pixel = image.at<Vec3b>(Point(j, i));
            int b = max(0, static_cast<int>(pixel[0]) - threshold);
            int g = max(0, static_cast<int>(pixel[1]) - threshold);
            int r = max(0, static_cast<int>(pixel[2]) - threshold);
            int total = b + g + r;
            if (total < 50)
            {
                resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                continue;
            }
            double brate = static_cast<double>(b) / total;
            double grate = static_cast<double>(g) / total;
            double rrate = static_cast<double>(r) / total;

            if (rrate > 0.65 && grate < 0.2)
            { // 红色要严格一些
                resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
            else if (grate > 0.55)
            { // 绿色要宽容些
                resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
            }
            else if (brate > 0.6)
            {
                resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            }
            else if (rrate > 0.3 && grate > 0.25 && rrate - brate > 0.1 && grate - brate > 0.1)
            {
                resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 255);
            }
            else
            {
                auto lab = BGR2Lab(pixel);
                auto a = lab[1];
                auto b = lab[2];
                if (a < -10)
                {
                    resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                }
                else if (b > 10)
                {
                    resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 255);
                }
                else
                {
                    resultImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }
    }
    return resultImage;
}

cv::Mat PictureH61DX::filterLowBrightnessAndburrs(cv::Mat &image, int threshold)
{
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    auto lower = Scalar(0, 0, threshold);
    auto upper = Scalar(180, 255, 255);
    Mat mask;
    inRange(hsv_image, lower, upper, mask);

    // 对原图进行掩膜操作
    Mat result;
    bitwise_and(image, image, result, mask);

    auto span = max(3, GroupUtilH61DX::getSpan(result));
    if (span > 3)
    {
        // 进行开闭运算
        morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
        bitwise_and(image, image, result, mask);
    }
    return result;
}
