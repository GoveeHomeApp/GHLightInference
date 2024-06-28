// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import "InferenceModule.h"
#import <LibTorch-Lite/LibTorch-Lite.h>

// 640x640 is the default image size used in the export.py in the yolov5 repo to export the TorchScript model, 25200*85 is the model output size
const int input_width = 640;
const int input_height = 640;
const int base_op = 25200;
const int output_size = 25200*8;
const int output_size_682 = 25200*6;

@implementation InferenceModule {
    @protected torch::jit::mobile::Module _impl;
}

- (instancetype)initWithFileAtPath:(NSString *)filePath withNc:(NSInteger)nc {
//    &output_size = base_op*nc;
    self.nc = (int)nc;
    return [self initWithFileAtPath:filePath];
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)detectImage:(void*)imageBuffer {
    try {
        NSLog(@"log.f inputWidth:%d", input_width);
        NSLog(@"log.f inputHeight:%d", input_width);
        NSLog(@"log.f outputSize:%d", output_size);
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, input_height, input_width }, at::kFloat);
        c10::InferenceMode guard;
        CFTimeInterval startTime = CACurrentMediaTime();
        auto outputTuple = _impl.forward({ tensor }).toTuple();
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);

        auto outputTensor = outputTuple->elements()[0].toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        if (self.nc == 6) {
            for (int i = 0; i < output_size_682; i++) {
                [results addObject:@(floatBuffer[i])];
            }
        } else {
            for (int i = 0; i < output_size; i++) {
                [results addObject:@(floatBuffer[i])];
            }
        }
        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
