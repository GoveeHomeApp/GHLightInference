
/**
 * Created by linpeng on 2024/7/3.
 *
在 alignSingleImage 函数中，我们现在返回一个 bool 值表示对齐是否成功，并通过引用返回对齐质量。
对齐质量基于 ECC 算法的返回值计算。如果 ECC 失败，我们设置一个较低的对齐质量。
增加了 alignmentThreshold 配置项，用于判断对齐是否足够好。

  alignSingleImageContour 函数，使用基于轮廓的方法进行对齐

 */
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc/types_c.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include "common.hpp"
#if TARGET_OS_IPHONE
#define USE_OPENCL 0
#else
#define USE_OPENCL 1
#endif
#else
#define USE_OPENCL 1
#endif

#if USE_OPENCL
#include <opencv2/core/ocl.hpp>
#endif

#if defined(__ANDROID__) || TARGET_OS_IPHONE
#ifndef NDEBUG
#define DEBUG 1
#endif
#endif
using namespace cv;
using namespace std;

class EnhancedChristmasTreeAligner {
public:
    struct AlignmentConfig {
        bool useScaling;
        double scale;
        int blurSize;
        int thresholdValue;
        int maxFeatures;
        float ransacReprojThreshold;
        int eccIterations;
        double eccEpsilon;
        bool useOpenCL;
        double alignmentThreshold;
        int contourMethod;
        int contourApproximation;

        // 默认构造函数
        AlignmentConfig()
                : useScaling(true),
                  scale(0.7),
                  blurSize(3),
                  thresholdValue(0),
                  maxFeatures(1000),
                  ransacReprojThreshold(3.0),
                  eccIterations(50),
                  eccEpsilon(0.001),
                  useOpenCL(false),
                  alignmentThreshold(0.98),
                  contourMethod(cv::RETR_EXTERNAL),
                  contourApproximation(cv::CHAIN_APPROX_SIMPLE) {}
    };

    EnhancedChristmasTreeAligner(const AlignmentConfig &cfg = AlignmentConfig()) : config(cfg) {
#if USE_OPENCL
        if (config.useOpenCL && cv::ocl::haveOpenCL()) {
            cv::ocl::setUseOpenCL(true);
        }
#endif
    }

    cv::Mat alignImage(const cv::Mat &firstImage, const cv::Mat &image) {
        if (image.empty()) return image;
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat aligned = image.clone();
        double alignmentQuality;
        bool success = false;

        try {
            // 尝试使用特征点对齐方法
            success = alignSingleImage(firstImage, image, aligned, alignmentQuality);
            // 如果特征点对齐失败或质量不佳，尝试轮廓对齐方法
            if (!success || alignmentQuality < config.alignmentThreshold) {
                aligned = alignImgEcc(firstImage, image);
                success = true;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            LOGD(LOG_TAG, "Aligning image took %f seconds", diff.count());
        } catch (const std::exception &e) {
            LOGE(LOG_TAG, "Error aligning image  e =  %s", e.what());
            aligned = alignImgEcc(firstImage, image);
        }
        return aligned;
    }

private:
    AlignmentConfig config;

    cv::Mat preprocess(const cv::Mat &input) {
        cv::Mat processed;

        if (config.useScaling) {
            cv::resize(input, processed, cv::Size(), config.scale, config.scale);
        } else {
            processed = input.clone();
        }

        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
        if (config.blurSize > 1) {
            cv::GaussianBlur(processed, processed, cv::Size(config.blurSize, config.blurSize), 0);
        }
        if (config.thresholdValue > 0) {
            cv::threshold(processed, processed, config.thresholdValue, 255, cv::THRESH_BINARY);
        }
        return processed;
    }

    bool alignSingleImage(const cv::Mat &reference, const cv::Mat &target, cv::Mat &result,
                          double &alignmentQuality) {
        cv::Mat refProcessed = preprocess(reference);
        cv::Mat targetProcessed = preprocess(target);

        // 特征点检测和匹配
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create(config.maxFeatures);
        std::vector<cv::KeyPoint> keypointsRef, keypointsTarget;
        cv::Mat descriptorsRef, descriptorsTarget;

#if USE_OPENCL
        if (config.useOpenCL && cv::ocl::haveOpenCL()) {
            LOGE(LOG_TAG,
                 "---------------------------> 使用 OpenCL 加速特征检测和计算 <--------------------------- ");
            // 使用 OpenCL 加速特征检测和计算
            cv::UMat uRefProcessed = refProcessed.getUMat(cv::ACCESS_READ);
            cv::UMat uTargetProcessed = targetProcessed.getUMat(cv::ACCESS_READ);
            cv::UMat uDescriptorsRef, uDescriptorsTarget;

            orb->detectAndCompute(uRefProcessed, cv::noArray(), keypointsRef, uDescriptorsRef);
            orb->detectAndCompute(uTargetProcessed, cv::noArray(), keypointsTarget,
                                  uDescriptorsTarget);

            uDescriptorsRef.copyTo(descriptorsRef);
            uDescriptorsTarget.copyTo(descriptorsTarget);
        } else {
#endif
            orb->detectAndCompute(refProcessed, cv::noArray(), keypointsRef, descriptorsRef);
            orb->detectAndCompute(targetProcessed, cv::noArray(), keypointsTarget,
                                  descriptorsTarget);
#if USE_OPENCL
        }
#endif

        // 使用 Brute-Force matcher 进行特征匹配
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
                "BruteForce-Hamming");
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptorsRef, descriptorsTarget, knnMatches, 2);

        // 应用比率测试来筛选好的匹配点
        std::vector<cv::DMatch> goodMatches;
        const float ratioThresh = 0.75f;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }
        // 检查是否有足够的好匹配点
        if (goodMatches.size() < 10) {
            LOGD(LOG_TAG, "Not enough good matches found. Matches : %d ", goodMatches.size());
            return false;
        }

        std::vector<cv::Point2f> pointsRef, pointsTarget;
        for (const auto &match: goodMatches) {
            pointsRef.push_back(keypointsRef[match.queryIdx].pt);
            pointsTarget.push_back(keypointsTarget[match.trainIdx].pt);
        }

        // 使用RANSAC估计变换矩阵
        std::vector<uchar> inliersMask;
        cv::Mat H = cv::findHomography(pointsTarget, pointsRef, cv::RANSAC,
                                       config.ransacReprojThreshold, inliersMask);

        // 计算内点比例
        float inlierRatio = static_cast<float>(cv::sum(inliersMask)[0]) / inliersMask.size();
        if (inlierRatio < 0.5f) {
            LOGD(LOG_TAG, "Inlier ratio too low. Ratio: %f ", inlierRatio);
            return false;
        }
        Mat targetGray, warpedTargetGray, referenceGray, warpedTarget;
        cvtColor(target, targetGray, CV_BGR2GRAY);
        cvtColor(reference, referenceGray, CV_BGR2GRAY);
        cv::warpPerspective(target, warpedTarget, H, target.size());
        cv::Mat warpMatrix = cv::Mat::eye(3, 3, CV_32F);

        // 使用ECC细化对齐
        try {
            cvtColor(warpedTarget, warpedTargetGray, CV_BGR2GRAY);
            // 确保图像类型一致
            double ecc = cv::findTransformECC(
                    referenceGray, warpedTargetGray, warpMatrix, cv::MOTION_HOMOGRAPHY,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                     config.eccIterations, config.eccEpsilon)
            );
            LOGW(LOG_TAG, "使用ECC细化对齐 ecc = %f ", ecc);
            alignmentQuality = ecc;
            // 检查对齐质量
            if (alignmentQuality < config.alignmentThreshold) {
                return false;
            }
            warpPerspective(warpedTarget, result, warpMatrix, warpedTarget.size(),
                            INTER_LINEAR + WARP_INVERSE_MAP);
        } catch (cv::Exception &e) {
            LOGE(LOG_TAG, "使用ECC细化对齐  e =  %s", e.what());
            warpMatrix = H;
            alignmentQuality = 0.5;
        }
        return true;
    }

    /**
     * 对齐图片
     */
    cv::Mat alignImgEcc(const Mat &src, const Mat &trans) {
        if (src.empty()) {
            return trans;
        }
        if (trans.empty()) {
            return src;
        }
        LOGD(LOG_TAG, "===========alignImgEcc===========");
        Mat alignedImg;
        try {
            Mat warp_matrix = Mat::eye(3, 3, CV_32F);

            // 降低图像分辨率
            // 创建掩膜，指定搜索区域
            Mat mask = Mat::zeros(trans.size(), CV_8UC1);

            Rect2i searchRegion = Rect(120, 100, 400, 400); // x, y, width, height

            rectangle(mask, searchRegion, Scalar::all(255), FILLED);
            //对齐精度
            double termination_eps2 = 1e-4;
            int number_of_iterations2 = 60;
            Mat im1Src, im2Trans;
            // 转换为灰度图
            cvtColor(src, im1Src, CV_BGR2GRAY);//CV_BGR2GRAY

            cvtColor(trans, im2Trans, CV_BGR2GRAY);

            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations2,
                                  termination_eps2);
            double ecc = findTransformECC(im1Src, im2Trans, warp_matrix, MOTION_HOMOGRAPHY,
                                          criteria, mask);//, mask
            LOGW(LOG_TAG, "alignImgEcc ecc = %f ", ecc);
            warpPerspective(trans, alignedImg, warp_matrix, trans.size(),
                            INTER_LINEAR + WARP_INVERSE_MAP);
            return alignedImg;
        } catch (cv::Exception &e) {
            LOGE(LOG_TAG, "alignImgEcc , e = %s", e.what());
            return trans;
        }
    }

    bool alignSingleImageContour(const cv::Mat &reference, const cv::Mat &target, cv::Mat &result,
                                 double &alignmentQuality) {
        cv::Mat refProcessed = preprocess(reference);
        cv::Mat targetProcessed = preprocess(target);

        std::vector<std::vector<cv::Point>> contoursRef, contoursTarget;
        cv::findContours(refProcessed, contoursRef, config.contourMethod,
                         config.contourApproximation);
        cv::findContours(targetProcessed, contoursTarget, config.contourMethod,
                         config.contourApproximation);

        if (contoursRef.empty() || contoursTarget.empty()) {
            std::cerr << "No contours found in one of the images" << std::endl;
            return false;
        }

        // 找到最大的轮廓（假设是圣诞树）
        auto maxContourRef = *std::max_element(contoursRef.begin(), contoursRef.end(),
                                               [](const std::vector<cv::Point> &c1,
                                                  const std::vector<cv::Point> &c2) {
                                                   return cv::contourArea(c1) < cv::contourArea(c2);
                                               });
        auto maxContourTarget = *std::max_element(contoursTarget.begin(), contoursTarget.end(),
                                                  [](const std::vector<cv::Point> &c1,
                                                     const std::vector<cv::Point> &c2) {
                                                      return cv::contourArea(c1) <
                                                             cv::contourArea(c2);
                                                  });

        // 计算轮廓的矩
        cv::Moments momentsRef = cv::moments(maxContourRef);
        cv::Moments momentsTarget = cv::moments(maxContourTarget);

        // 计算质心
        cv::Point2f centerRef(momentsRef.m10 / momentsRef.m00, momentsRef.m01 / momentsRef.m00);
        cv::Point2f centerTarget(momentsTarget.m10 / momentsTarget.m00,
                                 momentsTarget.m01 / momentsTarget.m00);

        // 计算平移
        cv::Point2f translation = centerRef - centerTarget;

        // 创建变换矩阵
        cv::Mat transMatrix = (cv::Mat_<float>(2, 3) << 1, 0, translation.x, 0, 1, translation.y);

        // 应用变换
        cv::warpAffine(target, result, transMatrix, reference.size());

        // 计算对齐质量（这里使用轮廓面积的比率作为简单的度量）
        double areaRef = cv::contourArea(maxContourRef);
        double areaTarget = cv::contourArea(maxContourTarget);
        alignmentQuality = std::min(areaRef, areaTarget) / std::max(areaRef, areaTarget);

        return true;
    }
};

//// 使用示例
////int main() {
////    try {
////        std::vector<cv::Mat> images;
////        for (int i = 0; i < 5; ++i) {
////            std::string filename = "tree_" + std::to_string(i) + ".jpg";
////            cv::Mat img = cv::imread(filename);
////            if (img.empty()) {
////                throw std::runtime_error("Failed to load image: " + filename);
////            }
////            images.push_back(img);
////        }
////
////        EnhancedChristmasTreeAligner::AlignmentConfig config;
////        config.useScaling = true;
////        config.scale = 0.5;
////        config.useOpenCL = true;
////        config.alignmentThreshold = 0.7;
////
////        EnhancedChristmasTreeAligner aligner(config);
////        std::vector<cv::Mat> alignedImages = aligner.alignImages(images);
////
////        for (size_t i = 0; i < alignedImages.size(); ++i) {
////            cv::imshow("Aligned Image " + std::to_string(i), alignedImages[i]);
////        }
////        cv::waitKey(0);
////    } catch (const std::exception &e) {
////        std::cerr << "An error occurred: " << e.what() << std::endl;
////        return -1;
////    }
////
////    return 0;
////}
