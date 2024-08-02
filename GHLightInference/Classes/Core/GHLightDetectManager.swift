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
    
    func prepostProcessor(config: ProcessorConfig) -> PrePostProcessor
}

public extension GHPytorchProtocol {
    
    func detection() { }
    
}

@objc open class GHLightDetectManager: NSObject {
    
    /// 单例
    @objc public private(set) static var instance = GHLightDetectManager()
    
    
}

extension GHLightDetectManager: GHPytorchProtocol {
    
    public func prepostProcessor(config: ProcessorConfig) -> PrePostProcessor {
        return PrePostProcessor(config: config)
    }
    
    public func detector() -> ObjectDetector {
        return ObjectDetector()
    }
}
