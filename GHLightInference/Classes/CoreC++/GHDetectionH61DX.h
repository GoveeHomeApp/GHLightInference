//
//  GHDetectionH61DX.h
//  GHLightInference
//
//  Created by luo on 2024/11/22.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface GHDetectionH61DX : NSObject

@property (nonatomic, assign) NSInteger icCount;

- (instancetype)initWithIcCount:(NSInteger)icCount;

/// 获取识别的灯效
- (NSArray<NSNumber *> *)getDetectionColors;

#if DEBUG

- (void)debugDetection:(UIImage *)originImage threshold:(UIImage *)thresholdImage callback:(void(^)(NSArray<UIImage *> *))callback;

#endif

@end

NS_ASSUME_NONNULL_END
