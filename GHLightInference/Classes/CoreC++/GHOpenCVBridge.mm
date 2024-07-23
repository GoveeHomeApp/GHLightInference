//
//  GHOpenCVBridge.mm
//  GHLightDetection
//
//  Created by sy on 2024/5/15.
//

#import "GHOpenCVBridge.h"
#include "features.hpp"
#include "sequence.hpp"

#import <iostream>
#import <stdlib.h>

const int red = -65536;
const int green = -16711936;
const int blue = -16776961;

static GHOpenCVBridge *manager = nil;

@interface GHOpenCVBridge()

@property (nonatomic, assign) NSInteger alignStep;

@end

@implementation GHOpenCVBridge

// 输出合集
vector<Mat> outMats;
float radiusCircle = 13;

// 无效回调测试帧
vector<Mat> emptyMats;

// pytorch识别返回结果集
vector<LightPoint> resLp;

+ (instancetype)shareManager {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        manager = [[self alloc] init];
    });
    return manager;
}

+ (id)allocWithZone:(struct _NSZone *)zone{
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        manager = [super allocWithZone:zone];
    });
    return manager;
}

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
    return manager;
}

- (nonnull id)mutableCopyWithZone:(nullable NSZone *)zone {
    return manager;
}

- (void)restAlignStep {
    self.alignStep = 0;
}

- (int)getMaxStep {
    return getMaxStepCnt();
}

- (void)createAllStepByIc:(NSInteger)icCount {
    initVector((int)icCount);
}

- (NSArray<NSArray<NSNumber *> *> *)getColorsByStep:(NSInteger)frameStep {
    NSMutableArray<NSNumber *> *redArray = [NSMutableArray array];
    NSMutableArray<NSNumber *> *greenArray = [NSMutableArray array];
    std::vector<std::vector<int>> colorMap;
    colorMap = getColors((int)frameStep);
    // 将std::vector转换为NSArray
    for (std::vector<int> subMap : colorMap) {
        int firstValue = subMap.front();
        int lastValue = subMap.back();
        if (lastValue == green) {
            [greenArray addObject: @(firstValue)];
        } else if (lastValue == red) {
            [redArray addObject: @(firstValue)];
        }
    }
    NSArray *resp = @[greenArray, redArray];
    return resp;
}

- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount {
    self.alignStep = stepCount;
    int ste = (int)stepCount;
    Mat aliMat = [self cvMatFromUIImage:image];
    cv::Mat imageBGR;
    cv::cvtColor(aliMat, imageBGR, cv::COLOR_RGBA2BGR);
    Mat resMat = alignResize(ste, imageBGR, outMats);
    cv::Mat resRGBA;
    cv::cvtColor(resMat, resRGBA, cv::COLOR_BGR2RGBA);
    return [self UIImageFromCVMat:resRGBA];
}

- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount rotation:(BOOL)isRotate {
    try {
        self.alignStep = stepCount;
        int ste = (int)stepCount;
        Mat aliMat = [self cvMatFromUIImage:image];
        Mat rotate = [self rotateImage90DegreesWithMat:aliMat rotate:ROTATE_90_CLOCKWISE];
        
        if (isRotate) {
            cv::Mat imageBGR;
            cv::cvtColor(rotate, imageBGR, cv::COLOR_RGBA2BGR);
            Mat resMat = alignResize(ste, imageBGR, outMats);
            cv::Mat resRGBA;
            cv::cvtColor(resMat, resRGBA, cv::COLOR_BGR2RGBA);
            return [self UIImageFromCVMat:resRGBA];
        } else {
            cv::Mat imageBGR;
            cv::cvtColor(aliMat, imageBGR, cv::COLOR_RGBA2BGR);
            Mat resMat = alignResize(ste, imageBGR, outMats);
            cv::Mat resRGBA;
            cv::cvtColor(resMat, resRGBA, cv::COLOR_BGR2RGBA);
            return [self UIImageFromCVMat:resRGBA];
        }
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
        throw exception;
    }
}

- (void)createLightPointArray:(NSArray *)poArray withBiz:(NSInteger)bizType {
    for (PredictObject *obj in poArray) {
        if (obj) {
            LightPoint ltp = LightPoint();
            ltp.tfScore = obj.score;
            Rect_<int> rect((int)obj.x, (int)obj.y, (int)obj.width, (int)obj.height);
            if ((int)bizType == TYPE_H682X) {
                ltp.with = rect.width;
                ltp.height = rect.height;
            } else {
                if (obj.type == 0) {
                    ltp.type = E_GREEN;
                } else if (obj.type == 1) {
                    ltp.type = E_RED;
                } else {
                    ltp.type = E_W;
                }
            }
            // 赋值中心点
            cv::Point center = cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
            ltp.position = center;
            ltp.tfRect = rect;
            resLp.push_back(ltp);
        }
    }
}

- (NSString *)caculateNumByStep:(NSInteger)stepCount bizType:(NSInteger)type {
    try {
        string jsonStr = "";
        jsonStr = sortStripByStep((int)stepCount, resLp, type, outMats);
        NSString * res = [NSString stringWithUTF8String:jsonStr.c_str()];
        return res;
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
        throw exception;
    }
}

- (void)clearAllresLp {
    resLp.clear();
}

- (UIImage *)showLastOutlet {
    Mat last = outMats.back();
    cv::Mat resRGBA;
    cv::cvtColor(last, resRGBA, cv::COLOR_BGR2RGBA);
    return [self UIImageFromCVMat:resRGBA];
}

// rgb to bgr通道
// openCV中图片的通道顺序与实际UIKit

/// rotate
- (Mat)rotateImage90DegreesWithMat:(Mat)image rotate:(int)rotateCode {
    Mat rotatedImage;
    rotate(image, rotatedImage, rotateCode);
    return rotatedImage;
}

// MARK: transfer mat and UIImage

- (cv::Point)convertUIImageCoordinateToCVMatPoint:(CGPoint)uiimageCoordinate imageSize:(CGSize)imgSize {
    CGFloat flippedY = imgSize.height - uiimageCoordinate.y;
    return cv::Point(uiimageCoordinate.x, flippedY);
}

/// mat from image
- (Mat)cvMatFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
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
- (Mat)cvMatGrayFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
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
-(UIImage *)UIImageGrayFromCVMat:(Mat)cvMat {
    Mat greyMat;
    cvtColor(cvMat, greyMat, COLOR_BGR2GRAY);
    return [self UIImageFromCVMat:greyMat];
}

-(UIImage *)UIImageFromCVMat:(Mat)cvMat {

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

