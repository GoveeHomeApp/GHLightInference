//
//  GHDetectionH61DX.mm
//  GHLightInference
//
//  Created by luo on 2024/11/22.
//

#import "GHDetectionH61DX.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#import "GHOpenCVBasic.h"
#include "DetectionH61DX.hpp"
#include "ColorCodingH61DX.hpp"

using namespace cv;
using namespace std;

@interface GHDetectionH61DX()
{
    DetectionH61DX _detection;
}
@end

@implementation GHDetectionH61DX

- (instancetype)initWithIcCount:(NSInteger)icCount {
    if (self = [super init]) {
        _icCount = icCount;
        _detection = DetectionH61DX(int(icCount));
    }
    return self;
}

- (NSArray<NSNumber *> *)getDetectionColors {
    NSMutableArray * result = [NSMutableArray array];
    auto colors = ColorCodingH61DX(int(_icCount)).getDetectionColors();
    for_each(colors.begin(), colors.end(), [&result](int item) {
        [result addObject:@(item)];
    });
    return result;
}

#if DEBUG

- (void)debugDetection:(UIImage *)originImage callback:(void (^)(NSArray<UIImage *> * _Nonnull))callback {
    Mat origin = [GHOpenCVBasic cvMatFromUIImage:originImage];
    _detection.debugDetection(origin, [&callback](std::vector<cv::Mat> arr) {
        NSMutableArray * result = [NSMutableArray array];
        for_each(arr.begin(), arr.end(), [&result](cv::Mat m){
            cv::Mat t;
            cv::cvtColor(m, t, cv::COLOR_BGR2RGBA);
            UIImage *img = [GHOpenCVBasic UIImageFromCVMat:t];
            if (img) {
                [result addObject:img];
            }
        });
        callback(result);
    });
}
#endif

@end
