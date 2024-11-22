//
//  GHOpenCVBasic.mm
//  GHLightDetection
//
//  Created by sy on 2024/5/15.
//

#import <iostream>
#import <stdlib.h>
#import "GHOpenCVBasic.h"

@interface GHOpenCVBasic()


@end

@implementation GHOpenCVBasic

/// rotate
+ (cv::Mat)rotateImage90DegreesWithMat:(cv::Mat)image rotate:(int)rotateCode {
    cv::Mat rotatedImage;
    rotate(image, rotatedImage, rotateCode);
    return rotatedImage;
}

// MARK: transfer mat and UIImage
+ (cv::Point)convertUIImageCoordinateToCVMatPoint:(CGPoint)uiimageCoordinate imageSize:(CGSize)imgSize {
    CGFloat flippedY = imgSize.height - uiimageCoordinate.y;
    return cv::Point(uiimageCoordinate.x, flippedY);
}

/// mat from image
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data, // Pointer to data
                                                    cols, // Width of bitmap
                                                    rows, // Height of bitmap
                                                    8, // Bits per component
                                                    cvMat.step[0], // Bytes per row
                                                    colorSpace, // Colorspace
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault); // Bitmap info flags
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    return cvMat;
}

// matgray from image
+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data, // Pointer to data
                                                    cols, // Width of bitmap
                                                    rows, // Height of bitmap
                                                    8, // Bits per component
                                                    cvMat.step[0], // Bytes per row
                                                    colorSpace, // Colorspace
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault); // Bitmap info flags
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    return cvMat;
 }

// 处理gray图像
+ (UIImage *)UIImageGrayFromCVMat:(cv::Mat)cvMat {
    cv::Mat greyMat;
    cvtColor(cvMat, greyMat, cv::COLOR_BGR2GRAY);
    return [self UIImageFromCVMat:greyMat];
}

+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols, //width
                                        cvMat.rows, //height
                                        8, //bits per component
                                        8 * cvMat.elemSize(), //bits per pixel
                                        cvMat.step[0], //bytesPerRow
                                        colorSpace, //colorspace
                                        kCGImageAlphaNoneSkipLast|kCGBitmapByteOrderDefault,// bitmap info
                                        provider, //CGDataProviderRef
                                        NULL, //decode
                                        false, //should interpolate
                                        kCGRenderingIntentDefault //intent
                                        );
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return finalImage;
 }

@end

