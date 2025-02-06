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

- (int)getMaxStep;

// 初始化灯序颜色
- (void)createAllStepByIc:(NSInteger)icCount;

// 返回当前涂鸦灯珠颜色（按greeArray redArray 返回）
- (NSArray<NSArray<NSNumber *> *> *)getColorsByStep:(NSInteger)frameStep;

// 带step的图片对齐 => 返回UIImage
- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount;
// 带step的图片对齐 => 返回UIImage 旋转90度
- (UIImage *)alignmentWithImage:(UIImage *)image step:(NSInteger)stepCount rotation:(BOOL)isRotate error:(void(^)(NSString* type))errorBlock;

// 识别完 生成vector<LightPoint>
- (void)createLightPointArray:(NSArray *)poArray withBiz:(NSInteger)bizType;

// 识别完位置 计算灯序
/// @param bizType: 业务类型 0-70XC 1-682X
- (NSString *)caculateNumByStep:(NSInteger)stepCount bizType:(NSInteger)type error:(void(^)(NSString* type))errorBlock;

// 获取last outlet
- (UIImage *)showLastOutlet;

- (void)clearAllresLp;

- (void)releaseOutProcess;

/* 分割算法实现方法 */


@end

NS_ASSUME_NONNULL_END
