//
//  SessionGenerator.swift
//  GHLightDetection
//
//  Created by sy on 2024/5/29.
//

import Foundation
import UIKit
import HandyJSON

public struct DetectionResult {
    /// 识别点 key: 序号 value 点 [x,y ...]
    public var points: [Int: [CGFloat]] = [:]
    /// 锚点 左上开始 右下结束
    public var anchorPoints: [[CGFloat]] = [[]]
    /// 识别放缩比例数组 [width, height ...]
    public var pixelScale: [CGFloat] = [640, 640]
    
    public var objectPoints: [LightQueueResult] = []
}

public struct DetectionEffectModel {
    /// 当前step
    public var frameStep: Int = 0
    public var colorDict: [UIColor: [Int]] = [:]
}

public class LightQueueBase: HandyJSON {
    
    public var lightPoints: [LightQueueResult] = []
    
    public var trapezoidalPoints: [LightQueueResult] = []
    
    required public init() { }
}

public class LightQueueResult: HandyJSON {
    
    public var x: Int = 0
    public var y: Int = 0
    public var startX: Int = 0
    public var startY: Int = 0
    public var endX: Int = 0
    public var endY: Int = 0
    public var index: Int = -1
    public var totalTfScore: Int = 0
    public var score: Int = 0
    public var tfScore: CGFloat = 0.0
    public var errorStatus:Int = 0
    public var pName: String = ""
    public var isBad: Bool = false
    
    required public init() { }
}

// 每次启动识别Session唯一标识
public class SessionGenerator {
    
    public private(set) static var instance = SessionGenerator()
    
    public var sku: String = ""
    
    public func newSessionId() -> String {
        UUID().uuidString
    }
    
    public func maxStepCountByIc(icCount: Int) -> Int {
        return Int(round(log2(Double(icCount))))
    }
    
}
