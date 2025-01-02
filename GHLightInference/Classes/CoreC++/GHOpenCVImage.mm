//
//  GHOpenCVImage.m
//  GHLightInference
//
//  Created by luo on 2024/11/22.
//

#import "GHOpenCVImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#import "GHOpenCVBasic.h"

using namespace cv;

@interface GHOpenCVImage()
{
    Mat _imageMat;
    cv::Size _kernelSize;
}
@end

@implementation GHOpenCVImage

- (void)dealloc {
    _imageMat.release();
}

- (instancetype)initWithImage:(UIImage *)image
{
    self = [super init];
    if (self) {
        _originImage = image;
        _kernelSize = cv::Size(3, 3);
        _imageMat = [GHOpenCVBasic cvMatFromUIImage:image];
    }
    return self;
}

- (UIImage *)toImage {
    if (_imageMat.empty()) {
        return _originImage;
    }
    return [GHOpenCVBasic UIImageFromCVMat:_imageMat];
}

- (GHOpenCVImage *)clone {
    GHOpenCVImage *copy = [[GHOpenCVImage alloc] init];
    copy.originImage = _originImage;
    copy->_kernelSize = _kernelSize;
    copy->_imageMat = _imageMat;
    return copy;
}

- (void)restore {
    if (_originImage) {
        _imageMat = [GHOpenCVBasic cvMatFromUIImage:_originImage];
    }
}

- (void)updateKernelSize:(int)size {
    _kernelSize = cv::Size(size, size);
}

- (void)cvtGrayColor {
    cvtColor(_imageMat, _imageMat, COLOR_BGR2GRAY);
}

- (void)gaussianBlur {
    GaussianBlur(_imageMat, _imageMat, _kernelSize, 0);
}

- (void)erosion:(NSInteger)times {
    if (times <= 0) { return; }
    Mat kernel = getStructuringElement(MORPH_RECT, _kernelSize);
    while (times--) {
        erode(_imageMat, _imageMat, kernel);
    }
}

- (void)dilating:(NSInteger)times {
    if (times <= 0) { return; }
    Mat kernel = getStructuringElement(MORPH_RECT, _kernelSize);
    while (times--) {
        dilate(_imageMat, _imageMat, kernel);
    }
}

- (void)morphologyExOpen:(NSInteger)times {
    if (times <= 0) { return; }
    Mat kernel = getStructuringElement(MORPH_RECT, _kernelSize);
    while (times--) {
        morphologyEx(_imageMat, _imageMat, MORPH_OPEN, kernel);
    }
}

- (void)morphologyExClose:(NSInteger)times {
    if (times <= 0) { return; }
    Mat kernel = getStructuringElement(MORPH_RECT, _kernelSize);
    while (times--) {
        morphologyEx(_imageMat, _imageMat, MORPH_CLOSE, kernel);
    }
}

- (void)subtract:(GHOpenCVImage *)other {
    cv::subtract(_imageMat, other->_imageMat, _imageMat);
}

- (void)originSubtractNow {
    if (!self.originImage) { return; }
    Mat origin = [GHOpenCVBasic cvMatFromUIImage:self.originImage];
    cv::subtract(origin, _imageMat, _imageMat);
}

- (void)bitwise_not {
    cv::bitwise_not(_imageMat, _imageMat);
}

- (void)bitwise_and:(GHOpenCVImage *)other {
    cv::bitwise_and(_imageMat, other->_imageMat, _imageMat);
}

- (void)bitwise_or:(GHOpenCVImage *)other {
    cv::bitwise_or(_imageMat, other->_imageMat, _imageMat);
}

- (void)bitwise_xor:(GHOpenCVImage *)other {
    cv::bitwise_xor(_imageMat, other->_imageMat, _imageMat);
}

@end
