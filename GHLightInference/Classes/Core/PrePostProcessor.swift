//// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import UIKit

public struct Prediction {
    public let classIndex: Int
    public let score: Float
    public let rect: CGRect
    
    public let x: Int
    public let y: Int
    public let w: Int
    public let h: Int
}

public struct ProcessorConfig {
    public var inputWidth = 640
    public var inputHeight = 640
    public var outputRow = 25200
    public var outputColumn = 8 // 默认红绿蓝
    public var threshold: Float = 0.01 // 目前非极大值抑制都是0.01
    public var nmsLimit = 1000
}

public struct ProcessorScaleConfig {
    public var scaleRate: CGFloat = 0.0
}

public class PrePostProcessor : NSObject {
    
    public init(config: ProcessorConfig) {
        self.config = config
    }
    
    public var config: ProcessorConfig = ProcessorConfig()
    
    // model input image size
    public var inputWidth: Int {
        config.inputWidth
    }
    public var inputHeight: Int {
        config.inputHeight
    }
    // model output is of size 25200*(num_of_class+5)
    public var outputRow: Int {
        config.outputRow
    } // as decided by the YOLOv5 model for input image of size 640*640
    public var outputColumn: Int {
        config.outputColumn
    } // left, top, right, bottom, score and 80 class probability
    public var threshold : Float {
        config.threshold
    } // score above which a detection is generated 非极大值抑制
    public var nmsLimit: Int {
        config.nmsLimit
    }// max number of detections
    
    // The two methods nonMaxSuppression and IOU below are from  https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
      Removes bounding boxes that overlap too much with other boxes that have
      a higher score.
      - Parameters:
        - boxes: an array of bounding boxes and their scores
        - limit: the maximum number of boxes that will be selected
        - threshold: used to decide whether boxes overlap too much
    */
    public func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {

      // Do an argsort on the confidence scores, from high to low.
      let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

      var selected: [Prediction] = []
      var active = [Bool](repeating: true, count: boxes.count)
      var numActive = active.count

      // The algorithm is simple: Start with the box that has the highest score.
      // Remove any remaining boxes that overlap it more than the given threshold
      // amount. If there are any boxes left (i.e. these did not overlap with any
      // previous boxes), then repeat this procedure, until no more boxes remain
      // or the limit has been reached.
      outer: for i in 0..<boxes.count {
        if active[i] {
          let boxA = boxes[sortedIndices[i]]
          selected.append(boxA)
          if selected.count >= limit { break }

          for j in i+1..<boxes.count {
            if active[j] {
              let boxB = boxes[sortedIndices[j]]
              if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                active[j] = false
                numActive -= 1
                if numActive <= 0 { break outer }
              }
            }
          }
        }
      }
      return selected
    }

    /**
      Computes intersection-over-union overlap between two bounding boxes.
    */
    public func IOU(a: CGRect, b: CGRect) -> Float {
        let areaA = a.width * a.height
        if areaA <= 0 { return 0 }

        let areaB = b.width * b.height
        if areaB <= 0 { return 0 }

        let intersectionMinX = max(a.minX, b.minX)
        let intersectionMinY = max(a.minY, b.minY)
        let intersectionMaxX = min(a.maxX, b.maxX)
        let intersectionMaxY = min(a.maxY, b.maxY)
        let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                             max(intersectionMaxX - intersectionMinX, 0)
        return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }

    public func outputsToNMSPredictions(outputs: [NSNumber], imgScaleX: Double, imgScaleY: Double, ivScaleX: Double, ivScaleY: Double, startX: Double, startY: Double) -> [Prediction] {
        var predictions = [Prediction]()
        for i in 0..<outputRow {
            if Float(truncating: outputs[i*outputColumn+4]) > threshold {
                // 中心点坐标以及宽高
                let x = Double(truncating: outputs[i*outputColumn])
                let y = Double(truncating: outputs[i*outputColumn+1])
                let w = Double(truncating: outputs[i*outputColumn+2])
                let h = Double(truncating: outputs[i*outputColumn+3])
                
                let left = imgScaleX * (x - w/2)
                let top = imgScaleY * (y - h/2)
                let right = imgScaleX * (x + w/2)
                let bottom = imgScaleY * (y + h/2)
                
                var max = Double(truncating: outputs[i*outputColumn+5])
                var cls = 0
                for j in 0 ..< outputColumn-5 {
                    if Double(truncating: outputs[i*outputColumn+5+j]) > max {
                        max = Double(truncating: outputs[i*outputColumn+5+j])
                        cls = j
                    }
                }
        
                let score = Float(truncating: outputs[i*outputColumn+4])
                let rect = CGRect(x: startX+ivScaleX*left, y: startY+top*ivScaleY, width: ivScaleX*(right-left), height: ivScaleY*(bottom-top))
                // 图片放缩比例 一定要知道！！！重点 不然找的位置都是错的 => 按现有放缩比例
                let prediction = Prediction(classIndex: cls, score: Float(truncating: outputs[i*outputColumn+4]), rect: rect, x: Int(x - w/2)*2, y: Int((y - h/2)/2*3), w: Int(w*2), h: Int(h/2*3))
                if score > 0.1 {
                    predictions.append(prediction)
                }
            }
        }

        return nonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: threshold)
    }
    
    public func originOutputsToNMSPredictions(outputs: [NSNumber], imgScaleX: Double, imgScaleY: Double, ivScaleX: Double, ivScaleY: Double, startX: Double, startY: Double) -> [Prediction] {
        var predictions = [Prediction]()
        for i in 0..<outputRow {
            if Float(truncating: outputs[i*outputColumn+4]) > threshold {
                let x = Double(truncating: outputs[i*outputColumn])
                let y = Double(truncating: outputs[i*outputColumn+1])
                let w = Double(truncating: outputs[i*outputColumn+2])
                let h = Double(truncating: outputs[i*outputColumn+3])
                
                let left = imgScaleX * (x - w/2)
                let top = imgScaleY * (y - h/2)
                let right = imgScaleX * (x + w/2)
                let bottom = imgScaleY * (y + h/2)
                
                var max = Double(truncating: outputs[i*outputColumn+5])
                var cls = 0
                for j in 0 ..< outputColumn-5 {
                    if Double(truncating: outputs[i*outputColumn+5+j]) > max {
                        max = Double(truncating: outputs[i*outputColumn+5+j])
                        cls = j
                    }
                }

                let rect = CGRect(x: startX+ivScaleX*left, y: startY+top*ivScaleY, width: ivScaleX*(right-left), height: ivScaleY*(bottom-top))
                let prediction = Prediction(classIndex: cls, score: Float(truncating: outputs[i*outputColumn+4]), rect: rect, x: Int(x), y: Int(y), w: Int(w), h: Int(h))
                predictions.append(prediction)
            }
        }

        return nonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: threshold)
    }

    public func cleanDetection(imageView: UIImageView) {
        if let layers = imageView.layer.sublayers {
            for layer in layers {
                if layer is CATextLayer {
                    layer.removeFromSuperlayer()
                }
            }
            for view in imageView.subviews {
                view.removeFromSuperview()
            }
        }
    }

    public func showDetection(imageView: UIImageView, nmsPredictions: [Prediction], classes: [String]) {
        
        debugPrint("Total object \(nmsPredictions.count)")
        
        for pred in nmsPredictions {
            let index = classes[pred.classIndex]
            switch index {
            case "red":
                let bbox = UIView(frame: pred.rect)
                bbox.backgroundColor = UIColor.clear
                bbox.layer.borderColor = UIColor.red.cgColor
                bbox.layer.borderWidth = 1
                if pred.score > 0.20 {
                    imageView.addSubview(bbox)
                }
            case "green":
                let bbox = UIView(frame: pred.rect)
                bbox.backgroundColor = UIColor.clear
                bbox.layer.borderColor = UIColor.green.cgColor
                bbox.layer.borderWidth = 1
                if pred.score > 0.20 {
                    imageView.addSubview(bbox)
                }
            default:
                break
            }
        }
    }
    
    public func showPreDetection(view: UIView, nmsPredictions: [Prediction], classes: [String], _ isHori: Bool = false, _ minus: CGFloat = 0.00) {
        debugPrint("log.p Total object \(nmsPredictions.count)")
        debugPrint("log.p view frame is \(view.frame)")
        let oriWidth = view.frame.size.width
        let oriHeight = view.frame.size.height
        print("log.p ===== minus \(minus)")
        view.subviews.map { $0.removeFromSuperview() }
        for pred in nmsPredictions {
            let index = classes[pred.classIndex]
            switch index {
            case "red":
                let bbox = UIView(frame: pred.rect)
                if isHori {
                    // 横屏坐标转换 注意x要乘以放缩比例
                    let transFrame = self.transferFrameToHori(view: view, origin: pred.rect, minus)
                    bbox.frame = CGRect(x: transFrame.origin.x, y: transFrame.origin.y, width: 10, height: 10)
                } else {
                    bbox.frame = CGRect(x: pred.rect.origin.x, y: pred.rect.origin.y, width: 10, height: 10)
                }
                bbox.backgroundColor = UIColor.blue.withAlphaComponent(0.8)
                bbox.layer.cornerRadius = 5
                bbox.layer.masksToBounds = true
                if pred.score > 0.20 {
                    view.addSubview(bbox)
                }
            case "green":
                let bbox = UIView(frame: pred.rect)
                if isHori {
                    // 横屏坐标转换 这个地方的位置要乘以宽高变换比例
                    let transFrame = self.transferFrameToHori(view: view, origin: pred.rect, minus)
                    bbox.frame = CGRect(x: transFrame.origin.x, y: transFrame.origin.y, width: 10, height: 10)
                } else {
                    bbox.frame = CGRect(x: pred.rect.origin.x, y: pred.rect.origin.y, width: 10, height: 10)
                }
                bbox.backgroundColor = UIColor.blue.withAlphaComponent(0.8)
                bbox.layer.cornerRadius = 5
                bbox.layer.masksToBounds = true
                if pred.score > 0.20 {
                    view.addSubview(bbox)
                }
            default:
                break
            }
        }
        view.clipsToBounds = true
    }
    
    public func transferFrameToHori(view: UIView, origin: CGRect, _ minus: CGFloat = 0.0) -> CGRect {
        var result = CGRect.zero
        let oriWidth = view.frame.size.width
        let oriHeight = view.frame.size.height
        let originX = origin.origin.y*oriHeight/oriWidth + (view.frame.origin.x*oriHeight/oriWidth) - minus
        var originY = origin.origin.x*oriHeight/oriWidth
        let backCenterY = view.frame.size.height/2
        // 横竖正确，需要沿页面中心X轴镜像翻转
        originY = abs(originY-2*backCenterY) - minus
        result = CGRect(x: originX, y: originY, width: origin.size.height, height: origin.size.width)
        return result
    }

}

class CrossDiagonalView: UIView {

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }

        // 第一条对角线：左下到右上
        let startPoint1 = CGPoint(x: rect.minX, y: rect.maxY)
        let endPoint1 = CGPoint(x: rect.maxX, y: rect.minY)

        // 第二条对角线：左上到右下
        let startPoint2 = CGPoint(x: rect.minX, y: rect.minY)
        let endPoint2 = CGPoint(x: rect.maxX, y: rect.maxY)

        // 设置线条属性
        context.setLineWidth(1.0)
        context.setStrokeColor(UIColor.cyan.cgColor)

        // 绘制第一条对角线
        context.move(to: startPoint1)
        context.addLine(to: endPoint1)
        context.strokePath()

        // 清除路径，以便重新开始绘制第二条对角线
        context.move(to: startPoint2)
        context.addLine(to: endPoint2)
        context.strokePath()
    }
}
