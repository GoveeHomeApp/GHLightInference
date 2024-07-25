//
// Created by linpeng on 2024/7/3.
//
#include <opencv2/opencv.hpp>
//#include <opencv2/core/ocl.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

class AcceleratedImageAligner {
private:
    std::vector<cv::Mat> images;
    cv::Mat referenceImage;
    int maxIterations;
    double terminationEps;
    std::vector<double> scaleFactors;
    bool useOpenCL;

    cv::Mat alignSingleImage(const cv::Mat &image) {
        cv::Mat currentAligned = image.clone();
        cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);

        for (double scale: scaleFactors) {
            cv::Mat small_image, small_reference;
            cv::resize(currentAligned, small_image, cv::Size(), scale, scale, cv::INTER_LINEAR);
            cv::resize(referenceImage, small_reference, cv::Size(), scale, scale, cv::INTER_LINEAR);

            cv::Mat gray_image, gray_reference;
            cv::cvtColor(small_image, gray_image, cv::COLOR_BGR2GRAY);
            cv::cvtColor(small_reference, gray_reference, cv::COLOR_BGR2GRAY);

            cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                      maxIterations, terminationEps);

            try {
                if (useOpenCL) {
//                    cv::UMat gray_image_umat = gray_image.getUMat(cv::ACCESS_RW);
//                    cv::UMat gray_reference_umat = gray_reference.getUMat(cv::ACCESS_RW);
//                    cv::UMat warp_matrix_umat = warp_matrix.getUMat(cv::ACCESS_RW);
//
//                    cv::findTransformECC(gray_reference_umat, gray_image_umat, warp_matrix_umat,
//                                         cv::MOTION_AFFINE, criteria);
//
//                    warp_matrix_umat.copyTo(warp_matrix);
                } else {
                    cv::findTransformECC(gray_reference, gray_image, warp_matrix,
                                         cv::MOTION_AFFINE, criteria);
                }

                warp_matrix.at<float>(0, 2) /= scale;
                warp_matrix.at<float>(1, 2) /= scale;

                cv::warpAffine(currentAligned, currentAligned, warp_matrix, currentAligned.size(),
                               cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
            }
//            catch (cv::Exception &e) {
//                std::cerr << "ECC alignment failed at scale " << scale << ": " << e.what()
//                          << std::endl;
//                // Continue to the next scale
//            }
            catch (...) {
                std::cerr << "ECC alignment failed at scale " << scale << std::endl;
            }
        }

        return currentAligned;
    }

    double assessAlignmentQuality(const cv::Mat &aligned, const cv::Mat &reference) {
        cv::Mat grayAligned, grayReference;
        cv::cvtColor(aligned, grayAligned, cv::COLOR_BGR2GRAY);
        cv::cvtColor(reference, grayReference, cv::COLOR_BGR2GRAY);

        cv::Mat diff;
        cv::absdiff(grayAligned, grayReference, diff);
        return 1.0 - (cv::mean(diff)[0] / 255.0);
    }

public:
    AcceleratedImageAligner(int maxIter = 50, double termEps = 1e-5)
            : maxIterations(maxIter), terminationEps(termEps) {
        scaleFactors = {0.25, 0.5, 0.75, 1.0};  // Multi-scale approach
        useOpenCL = false;
        if (useOpenCL) {
//            cv::ocl::setUseOpenCL(true);
            std::cout << "OpenCL is available. Using GPU acceleration." << std::endl;
        } else {
            std::cout << "OpenCL is not available. Using CPU processing." << std::endl;
        }
    }

    void addImage(const cv::Mat &image) {
        images.push_back(image);
    }

    void setReferenceImage(const cv::Mat &refImage) {
        referenceImage = refImage;
    }

    std::vector<cv::Mat> alignImages() {
        std::vector<cv::Mat> alignedImages;
        std::vector<double> qualities;

        for (const auto &img: images) {
            cv::Mat aligned = alignSingleImage(img);
            double quality = assessAlignmentQuality(aligned, referenceImage);
            alignedImages.push_back(aligned);
            qualities.push_back(quality);
        }

        // Output quality assessment results
        for (size_t i = 0; i < qualities.size(); ++i) {
            std::cout << "Image " << i << " alignment quality: " << qualities[i] << std::endl;
        }

        return alignedImages;
    }
};

// Usage example
//void processImages(const std::vector<cv::Mat>& inputImages) {
//    AcceleratedImageAligner aligner(30, 1e-5);
//    aligner.setReferenceImage(inputImages[0]);
//
//    for (size_t i = 1; i < inputImages.size(); ++i) {
//        aligner.addImage(inputImages[i]);
//    }
//
//    std::vector<cv::Mat> alignedImages = aligner.alignImages();
//
//    // Here you can do further processing, such as saving aligned images
//}
