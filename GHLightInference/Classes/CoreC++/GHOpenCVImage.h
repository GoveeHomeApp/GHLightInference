//
//  GHOpenCVImage.h
//  GHLightInference
//
//  Created by luo on 2024/11/22.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface GHOpenCVImage : NSObject

@property (nonatomic, strong, nonnull) UIImage * originImage;

- (instancetype)initWithImage:(UIImage * __nonnull)image;

- (UIImage * __nullable)toImage;

- (GHOpenCVImage *)clone;

/// 还原所有操作
- (void)restore;

- (void)updateKernelSize:(int)size;

/// 转为灰度图像
- (void)cvtGrayColor;

/// 高斯模糊
- (void)gaussianBlur;

/// 腐蚀
/// - Parameter times: 执行次数
- (void)erosion:(NSInteger)times;

/// 膨胀
/// - Parameter times: 执行次数
- (void)dilating:(NSInteger)times;

/// 开运算
/// - Parameter times: 执行次数
- (void)morphologyExOpen:(NSInteger)times;

/// 关运算
/// - Parameter times: 执行次数
- (void)morphologyExClose:(NSInteger)times;

/// 减去image
- (void)subtract:(GHOpenCVImage *)other;

/// 原始值减去当前的
- (void)originSubtractNow;

/// 按位取反
- (void)bitwise_not;

/// 按位与操作
- (void)bitwise_and:(GHOpenCVImage *)other;

/// 按位或操作
- (void)bitwise_or:(GHOpenCVImage *)other;

/// 按位异或操作
- (void)bitwise_xor:(GHOpenCVImage *)other;

@end

NS_ASSUME_NONNULL_END
