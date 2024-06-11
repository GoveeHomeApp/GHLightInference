//
//  GHLightDetection.swift
//  GHLightDetection
//
//  Created by yang song on 2024/5/4.
//

import Foundation

public protocol GHPytorchProtocol {
    
    func detection()
    
    func detector() -> ObjectDetector
    
    func prepostProcessor() -> PrePostProcessor
}

public extension GHPytorchProtocol {
    
    func detection() { }
    
}

@objc open class GHLightDetectManager: NSObject {
    
    /// 单例
    @objc public private(set) static var instance = GHLightDetectManager()
    
    
}

extension GHLightDetectManager: GHPytorchProtocol {
    
    public func prepostProcessor() -> PrePostProcessor {
        return PrePostProcessor()
    }
    
    public func detector() -> ObjectDetector {
        return ObjectDetector()
    }
    
    @objc public func detection() {
        let resizedImage = UIImage()
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        DispatchQueue.global().async {
            guard let outputs = self.detector().module.detect(image: &pixelBuffer) else {
                return
            }
        }
    }
}
