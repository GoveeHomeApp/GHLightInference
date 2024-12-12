#include "PictureH61DX.hpp"
#include "logger.hpp"

using namespace cv;
using namespace std;

namespace
{
    // 图片缩放目标宽度
    const int TARGET_WIDTH = 400;
}

cv::Mat PictureH61DX::processImage(cv::Mat& image) 
{
    if (image.empty())
    {
        return image.clone();
    }
    // 缩放图片
    Mat resizedImage = resizeWithAspectRatio(image, TARGET_WIDTH);
    // 提纯颜色
    Mat replacedImage = purifiedImage(resizedImage);
    // 开运算2次
    Mat openedImage;
    morphologyEx(replacedImage, openedImage, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 2);
    // 闭运算1次
    Mat closeImage;
    morphologyEx(openedImage, closeImage, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 1);
    return closeImage;
}

cv::Mat PictureH61DX::resizeWithAspectRatio(const cv::Mat& image, int target, int inter) {
    Size dim;
    int h = image.rows;
    int w = image.cols;

    if (w < h) {
        float r = static_cast<float>(target) / w;
        dim = Size(target, static_cast<int>(h * r));
    } else {
        float r = static_cast<float>(target) / h;
        dim = Size(static_cast<int>(w * r), target);
    }
    
    Mat resized;
    resize(image, resized, dim, 0, 0, inter);
    return resized;
}

cv::Mat PictureH61DX::purifiedImage(cv::Mat& image)
{
    auto resultImage = image.clone();
    if (image.empty())
    {
        return resultImage;
    }
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            Vec3b pixel = image.at<Vec3b>(Point(j, i));
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            int thresholdHigh = 100;
            int thresholdSimilar = 50;

            if ((r < thresholdHigh && g < thresholdHigh && b < thresholdHigh) ||
                (abs(r - g) < thresholdSimilar && abs(r - b) < thresholdSimilar && abs(g - b) < thresholdSimilar)) {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
            }
            else if (r > g + thresholdSimilar && r > b + thresholdSimilar) {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 255);
            }
            else if (g > r + thresholdSimilar && g > b + thresholdSimilar) {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(0, 255, 0);
            }
            else if (b > r + thresholdSimilar && b > g + thresholdSimilar) {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(255, 0, 0);
            }
            else if (abs(r - g) < thresholdSimilar && r > b + thresholdSimilar && g > b + thresholdSimilar) {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(0, 255, 255);
            }
            else {
                resultImage.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
            }
        }
    }
    return resultImage;
}
