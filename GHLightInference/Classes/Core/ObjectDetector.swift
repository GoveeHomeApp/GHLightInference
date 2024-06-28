// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import UIKit

public class ObjectDetector {
    
    public var sku: String = ""
    public var dimension: String = "3D" //默认3D
    
    public lazy var module: InferenceModule = {
        if let path = NSSearchPathForDirectoriesInDomains(.libraryDirectory, .userDomainMask, true).first, let module = InferenceModule(fileAtPath: path+"/Detection/\(sku)/\(dimension)/best.torchscript.ptl", withNc: classes.count) {
            return module
        } else if let path = NSSearchPathForDirectoriesInDomains(.libraryDirectory, .userDomainMask, true).first, let module = InferenceModule(fileAtPath: path+"/Detection/\(sku)/3D/best.torchscript.ptl", withNc: classes.count) {
            // 给2D兜底
            return module
        } else if let filePath = Bundle.main.path(forResource: "best.torchscript", ofType: "ptl"),
                  let module = InferenceModule(fileAtPath: filePath, withNc: classes.count) {
            return module
        } else if let fp = Bundle.bundleResource(of: GHLightInference.self)?.path(forResource: "best.torchscript", ofType: "ptl"),
                  let module = InferenceModule(fileAtPath: fp, withNc: classes.count) {
            return module
        } else {
            fatalError("Failed to load model!")
        }
    }()
    
    public lazy var classes: [String] = {
        if sku.hasPrefix("H682") {
            return ["light"]
        } else {
            return ["green",
                    "red",
                    "blue"]
        }
    }()
}
