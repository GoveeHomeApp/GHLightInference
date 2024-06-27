//
//  PredictObject.h
//  GHLightDetection
//
//  Created by sy on 2024/5/31.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface PredictObject : NSObject

@property (nonatomic, assign) NSInteger x;
@property (nonatomic, assign) NSInteger y;
@property (nonatomic, assign) NSInteger width;
@property (nonatomic, assign) NSInteger height;

@property (nonatomic, assign) NSInteger type;

@property (nonatomic, assign) CGFloat score;

@property (nonatomic, assign) NSInteger lightId;

@end

NS_ASSUME_NONNULL_END
