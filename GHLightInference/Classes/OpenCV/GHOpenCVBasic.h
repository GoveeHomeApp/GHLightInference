//
//  GHOpenCVBasic.h
//  GHLightDetection
//
//  Created by sy on 2024/5/15.
//

#import <Foundation/Foundation.h>
#include <opencv2/opencv.hpp>

NS_ASSUME_NONNULL_BEGIN

@interface GHOpenCVBasic : NSObject

+ (cv::Mat)rotateImage90DegreesWithMat:(cv::Mat)image rotate:(int)rotateCode;
+ (cv::Point)convertUIImageCoordinateToCVMatPoint:(CGPoint)uiimageCoordinate imageSize:(CGSize)imgSize;
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image;
+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image;
+ (UIImage *)UIImageGrayFromCVMat:(cv::Mat)cvMat;
+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;

@end

NS_ASSUME_NONNULL_END
