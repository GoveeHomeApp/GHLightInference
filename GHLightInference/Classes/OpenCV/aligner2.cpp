
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
#include "common.hpp"
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
                  eccIterations(20),
                  eccEpsilon(0.001),
                  useOpenCL(false),
                  alignmentThreshold(0.98),
                  contourMethod(RETR_EXTERNAL),
                  contourApproximation(CHAIN_APPROX_SIMPLE) {}
    };

    // 提取图像中心三分之一区域
    Mat extractCenterThird(const Mat &input) {
        int centerX = input.cols / 2;
        int centerY = input.rows / 2;
        int width = input.cols / 3;
        int height = input.rows / 3;

        Rect centerROI(centerX - width / 2, centerY - height / 2, width, height);
        return input(centerROI).clone();
    }

    EnhancedChristmasTreeAligner(const AlignmentConfig &cfg = AlignmentConfig()) : config(cfg) {
#if USE_OPENCL
        if (config.useOpenCL && ocl::haveOpenCL()) {
            ocl::setUseOpenCL(true);
        }
#endif
    }

    Mat alignImage(const Mat &firstImage, const Mat &image, vector<Mat> &outMats) {
        if (image.empty()) return image;
        auto start = std::chrono::high_resolution_clock::now();

        Mat aligned;
        double alignmentQuality;
        bool success = false;

        try {
            // 尝试使用特征点对齐方法
//            success = alignSingleImage(firstImage, image, aligned, alignmentQuality);
            // 如果特征点对齐失败或质量不佳，尝试轮廓对齐方法
//            if (!success || alignmentQuality < config.alignmentThreshold) {
//                outMats.push_back(firstImage);
//                outMats.push_back(image);
            aligned = alignImgEcc(firstImage, image);
//                outMats.push_back(aligned);
//                success = true;
//            }
//            auto end = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<double> diff = end - start;
//            LOGD(LOG_TAG, "Aligning image took %f seconds", diff.count());
        } catch (...) {
//            LOGE(LOG_TAG, "Error aligning image  e =  %s", e.what());
//            aligned = alignImgEcc(firstImage, image);
            return image;
        }
        return aligned;
    }

private:
    AlignmentConfig config;

    Mat preprocess(const Mat &input) {
        Mat processed;

        if (config.useScaling) {
            resize(input, processed, Size(), config.scale, config.scale);
        } else {
            processed = input.clone();
        }

        cvtColor(processed, processed, COLOR_BGR2GRAY);
        if (config.blurSize > 1) {
            GaussianBlur(processed, processed, Size(config.blurSize, config.blurSize), 0);
        }
        if (config.thresholdValue > 0) {
            threshold(processed, processed, config.thresholdValue, 255, THRESH_BINARY);
        }
        return processed;
    }

//    bool alignSingleImage(const Mat &reference, const Mat &target, Mat &result,
//                          double &alignmentQuality) {
//        // 提取中心区域
//        Mat refCenter = extractCenterThird(reference);
//        Mat targetCenter = extractCenterThird(target);
//
//        // 预处理
//        Mat refProcessed = preprocess(refCenter);
//        Mat targetProcessed = preprocess(targetCenter);
//
//        // 特征点检测和匹配
//        Ptr <Feature2D> orb = ORB::create(config.maxFeatures);
//        std::vector <KeyPoint> keypointsRef, keypointsTarget;
//        Mat descriptorsRef, descriptorsTarget;
//
//#if USE_OPENCL
//        if (config.useOpenCL && ocl::haveOpenCL()) {
//            LOGE(LOG_TAG,
//                 "---------------------------> 使用 OpenCL 加速特征检测和计算 <--------------------------- ");
//            // 使用 OpenCL 加速特征检测和计算
//            UMat uRefProcessed = refProcessed.getUMat(ACCESS_READ);
//            UMat uTargetProcessed = targetProcessed.getUMat(ACCESS_READ);
//            UMat uDescriptorsRef, uDescriptorsTarget;
//
//            orb->detectAndCompute(uRefProcessed, noArray(), keypointsRef, uDescriptorsRef);
//            orb->detectAndCompute(uTargetProcessed, noArray(), keypointsTarget,
//                                  uDescriptorsTarget);
//
//            uDescriptorsRef.copyTo(descriptorsRef);
//            uDescriptorsTarget.copyTo(descriptorsTarget);
//        } else {
//#endif
//            orb->detectAndCompute(refProcessed, noArray(), keypointsRef, descriptorsRef);
//            orb->detectAndCompute(targetProcessed, noArray(), keypointsTarget,
//                                  descriptorsTarget);
//#if USE_OPENCL
//        }
//#endif
//
//        // 使用 Brute-Force matcher 进行特征匹配
//        Ptr <DescriptorMatcher> matcher = DescriptorMatcher::create(
//                "BruteForce-Hamming");
//        std::vector <std::vector<DMatch>> knnMatches;
//        matcher->knnMatch(descriptorsRef, descriptorsTarget, knnMatches, 2);
//
//        // 应用比率测试来筛选好的匹配点
//        std::vector <DMatch> goodMatches;
//        const float ratioThresh = 0.75f;
//        for (size_t i = 0; i < knnMatches.size(); i++) {
//            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
//                goodMatches.push_back(knnMatches[i][0]);
//            }
//        }
//        // 检查是否有足够的好匹配点
//        if (goodMatches.size() < 10) {
//            LOGD(LOG_TAG, "Not enough good matches found. Matches : %d ", goodMatches.size());
//            return false;
//        }
//
//        std::vector <Point2f> pointsRef, pointsTarget;
//        for (const auto &match: goodMatches) {
//            pointsRef.push_back(keypointsRef[match.queryIdx].pt);
//            pointsTarget.push_back(keypointsTarget[match.trainIdx].pt);
//        }
//
//        // 使用RANSAC估计变换矩阵
//        std::vector <uchar> inliersMask;
//        Mat H = findHomography(pointsTarget, pointsRef, RANSAC,
//                               config.ransacReprojThreshold, inliersMask);
//
//        // 计算内点比例
//        float inlierRatio = static_cast<float>(sum(inliersMask)[0]) / inliersMask.size();
//        if (inlierRatio < 0.5f) {
//            LOGD(LOG_TAG, "Inlier ratio too low. Ratio: %f ", inlierRatio);
//            return false;
//        }
//        Mat warpedTargetGray, referenceGray, warpedTarget;
//        cvtColor(refCenter, referenceGray, CV_BGR2GRAY);
//        warpPerspective(targetCenter, warpedTarget, H, target.size());
//        Mat warpMatrix = Mat::eye(3, 3, CV_32F);
//        LOGW(LOG_TAG, "完成特征匹配对齐 " );
//        // 使用ECC细化对齐
//        try {
//            cvtColor(warpedTarget, warpedTargetGray, CV_BGR2GRAY);
//
//            // 确保图像类型一致
//            double ecc = findTransformECC(
//                    referenceGray, warpedTargetGray, warpMatrix, MOTION_HOMOGRAPHY,
//                    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
//                                 config.eccIterations, config.eccEpsilon)
//            );
//            LOGW(LOG_TAG, "使用ECC细化对齐 ecc = %f ", ecc);
//            alignmentQuality = ecc;
//            // 检查对齐质量
//            if (alignmentQuality < config.alignmentThreshold) {
//                return false;
//            }
//            warpPerspective(warpedTarget, result, warpMatrix, warpedTarget.size(),
//                            INTER_LINEAR + WARP_INVERSE_MAP);
//        } catch (Exception &e) {
//            LOGE(LOG_TAG, "使用ECC细化对齐  e =  %s", e.what());
//            warpMatrix = H;
//            alignmentQuality = 0.5;
//        }
//        return true;
//    }

    /**
     * 对齐图片
     */
    cv::Mat alignImgEcc(const Mat &src, const Mat &trans, int motionTypeSet = MOTION_AFFINE) {
        if (src.empty()) {
            return trans;
        }
        if (trans.empty()) {
            return src;
        }
        Mat alignedImg;
        try {
            Mat warp_matrix;
            if (motionTypeSet == MOTION_AFFINE) {
                warp_matrix = Mat::eye(2, 3, CV_32F);
            } else if (motionTypeSet == MOTION_HOMOGRAPHY) {//MOTION_HOMOGRAPHY 耗时更久
                warp_matrix = Mat::eye(3, 3, CV_32F);
            } else {
                //MOTION_EUCLIDEAN
//                motionTypeSet = MOTION_EUCLIDEAN;
                warp_matrix = Mat::eye(2, 3, CV_32F);
            }
            // 降低图像分辨率
            // 创建掩膜，指定搜索区域
            Mat mask = Mat::zeros(trans.size(), CV_8UC1);

            Rect2i searchRegion = Rect(120, 100, 400, 400); // x, y, width, height

            rectangle(mask, searchRegion, Scalar::all(255), FILLED);
            //对齐精度
            double termination_eps2 = 0.0005;
            int number_of_iterations2 = 70;
            Mat im1Src, im2Trans;
            // 转换为灰度图
            cvtColor(src, im1Src, COLOR_BGR2GRAY);//CV_BGR2GRAY

            cvtColor(trans, im2Trans, COLOR_BGR2GRAY);

            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations2,
                                  termination_eps2);
            double ecc = findTransformECC(im1Src, im2Trans, warp_matrix, motionTypeSet,
                                          criteria, mask);
            LOGW(LOG_TAG, "alignImgEcc ecc = %f ", ecc);
            if (motionTypeSet == MOTION_HOMOGRAPHY) {
                warpPerspective(trans, alignedImg, warp_matrix, trans.size(),
                                INTER_LINEAR + WARP_INVERSE_MAP);
            } else {
                warpAffine(trans, alignedImg, warp_matrix, trans
                        .size(), INTER_LINEAR + WARP_INVERSE_MAP);
            }
            return alignedImg;
        } catch (...) {
            LOGE(LOG_TAG, "alignImgEcc error");
            return trans;
        }
    }

    bool alignSingleImageContour(const Mat &reference, const Mat &target, Mat &result,
                                 double &alignmentQuality) {
        Mat refProcessed = preprocess(reference);
        Mat targetProcessed = preprocess(target);

        std::vector<std::vector<Point>> contoursRef, contoursTarget;
        findContours(refProcessed, contoursRef, config.contourMethod,
                     config.contourApproximation);
        findContours(targetProcessed, contoursTarget, config.contourMethod,
                     config.contourApproximation);

        if (contoursRef.empty() || contoursTarget.empty()) {
            std::cerr << "No contours found in one of the images" << std::endl;
            return false;
        }

        // 找到最大的轮廓（假设是圣诞树）
        auto maxContourRef = *std::max_element(contoursRef.begin(), contoursRef.end(),
                                               [](const std::vector<Point> &c1,
                                                  const std::vector<Point> &c2) {
                                                   return contourArea(c1) < contourArea(c2);
                                               });
        auto maxContourTarget = *std::max_element(contoursTarget.begin(), contoursTarget.end(),
                                                  [](const std::vector<Point> &c1,
                                                     const std::vector<Point> &c2) {
                                                      return contourArea(c1) <
                                                             contourArea(c2);
                                                  });

        // 计算轮廓的矩
        Moments momentsRef = moments(maxContourRef);
        Moments momentsTarget = moments(maxContourTarget);

        // 计算质心
        Point2f centerRef(momentsRef.m10 / momentsRef.m00, momentsRef.m01 / momentsRef.m00);
        Point2f centerTarget(momentsTarget.m10 / momentsTarget.m00,
                             momentsTarget.m01 / momentsTarget.m00);

        // 计算平移
        Point2f translation = centerRef - centerTarget;

        // 创建变换矩阵
        Mat transMatrix = (Mat_<float>(2, 3) << 1, 0, translation.x, 0, 1, translation.y);

        // 应用变换
        warpAffine(target, result, transMatrix, reference.size());

        // 计算对齐质量（这里使用轮廓面积的比率作为简单的度量）
        double areaRef = contourArea(maxContourRef);
        double areaTarget = contourArea(maxContourTarget);
        alignmentQuality = std::min(areaRef, areaTarget) / std::max(areaRef, areaTarget);

        return true;
    }
};

//// 使用示例
////int main() {
////    try {
////        std::vector<Mat> images;
////        for (int i = 0; i < 5; ++i) {
////            std::string filename = "tree_" + std::to_string(i) + ".jpg";
////            Mat img = imread(filename);
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
////        std::vector<Mat> alignedImages = aligner.alignImages(images);
////
////        for (size_t i = 0; i < alignedImages.size(); ++i) {
////            imshow("Aligned Image " + std::to_string(i), alignedImages[i]);
////        }
////        waitKey(0);
////    } catch (const std::exception &e) {
////        std::cerr << "An error occurred: " << e.what() << std::endl;
////        return -1;
////    }
////
////    return 0;
////}
