//
//  GHOpenCVBridge.h
//  GHLightDetection
//
//  Created by sy on 2024/5/15.
//

#import <Foundation/Foundation.h>
#import "PredictObject.h"

NS_ASSUME_NONNULL_BEGIN

@interface GHOpenCVBridge : NSObject

+ (instancetype)shareManager;

// 重设step
- (void)restAlignStep;

// 初始化灯序颜色
- (void)createAllStepByIc:(NSInteger)icCount;

// 返回当前涂鸦灯珠颜色（按greeArray redArray 返回）
- (NSArray<NSArray<NSNumber *> *> *)getColorsByStep:(NSInteger)frameStep;

// 带step的图片对齐 => 返回UIImage
- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount;
// 带step的图片对齐 => 返回UIImage 旋转90度
- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount rotation:(BOOL)isRotate;

// 识别完 生成vector<LightPoint>
- (void)createLightPointArray:(NSArray *)poArray;

// 识别完位置 计算灯序
- (NSString *)caculateNumByStep:(NSInteger)stepCount;

// 获取last outlet
- (UIImage *)showLastOutlet;

@end

NS_ASSUME_NONNULL_END
