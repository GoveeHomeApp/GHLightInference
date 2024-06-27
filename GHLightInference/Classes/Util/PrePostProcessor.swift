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
    public var inpitHeight = 640
    public var outputRow = 25200
    public var outputColumn = 8 // 默认红绿蓝
    public var threshold: CGFloat = 0.01 // 目前非极大值抑制都是0.01
    public var nmsLimit = 1000
}

//public struct ProcessorScaleConfig {
//    
//    
//    
//}

public class PrePostProcessor : NSObject {
    
    static var config: ProcessorConfig = ProcessorConfig()
    
    // model input image size
    static public let inputWidth = 640
    static public let inputHeight = 640

    // model output is of size 25200*(num_of_class+5)
    static public let outputRow = 25200 // as decided by the YOLOv5 model for input image of size 640*640
    static public let outputColumn = 8 // left, top, right, bottom, score and 80 class probability
    static public let threshold : Float = 0.01 // score above which a detection is generated 非极大值抑制
    static public let nmsLimit = 1000 // max number of detections
    
    // The two methods nonMaxSuppression and IOU below are from  https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
      Removes bounding boxes that overlap too much with other boxes that have
      a higher score.
      - Parameters:
        - boxes: an array of bounding boxes and their scores
        - limit: the maximum number of boxes that will be selected
        - threshold: used to decide whether boxes overlap too much
    */
    static public func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {

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
    static public func IOU(a: CGRect, b: CGRect) -> Float {
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

    static public func outputsToNMSPredictions(outputs: [NSNumber], imgScaleX: Double, imgScaleY: Double, ivScaleX: Double, ivScaleY: Double, startX: Double, startY: Double) -> [Prediction] {
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

    static public func cleanDetection(imageView: UIImageView) {
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

    static public func showDetection(imageView: UIImageView, nmsPredictions: [Prediction], classes: [String]) {
        
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
            case "blue":
//                let bbox = UIView(frame: pred.rect)
//                bbox.backgroundColor = UIColor.clear
//                bbox.layer.borderColor = UIColor.blue.cgColor
//                bbox.layer.borderWidth = 1
//                if pred.score > 0.1 {
//                    imageView.addSubview(bbox)
//                }
                break
            default:
                break
            }
        }
    }

}
