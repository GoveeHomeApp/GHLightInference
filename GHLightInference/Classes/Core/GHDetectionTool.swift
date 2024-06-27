//
//  GHDetectionTool.swift
//  GHLightDetection
//
//  Created by sy on 2024/6/3.
//

import Foundation
import AVFoundation
import CoreImage
import Photos

// AVFoundation 必须是OC类
public class GHDetectionTool: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    public var initializeFinishNotice: (() -> Void)?
    
    public var finishFrameNotice: ((Bool) -> Void)?
    // 取帧Notice（每一帧需要的灯效， 开始成功就返回第一帧）
    public var frameNotice: ((DetectionEffectModel) -> Void)?
    // 完成Notice
    public var doneNotice: ((DetectionResult?) -> Void)?
    
    // 开始 & 重新开始 带有transaction
    public private(set) var startHandler: ((String) -> Void)?
    // 中断
    public private(set) var interruptHandler: (() -> Void)?
    // 取帧（第几步 ps:第几次取帧 从序号1开始）
    public private(set) var frameHandler:((Int) -> Void)?
    
    private var capFinishHandler: (() -> Void)?
    
    // 识别transaction
    private var transaction: String? {
        didSet {
            if let t = transaction {
                
            } else {
                self.preImageArray.removeAll()
                self.afterImgArray.removeAll()
            }
        }
    }
    
    public var sku: String = ""
    public var ic: Int = 0
    var bizType: Int = 0
    
    var captureSession: AVCaptureSession!
    var videoOutput: AVCaptureVideoDataOutput!
    // 直接拿Layer
    public private(set) var previewLayer: AVCaptureVideoPreviewLayer?
    
    var needGetFrame: Bool = false
    var preImageArray: [UIImage] = []
    var afterImgArray: [UIImage] = []
    
    private var inferencer = ObjectDetector.instance
    private var prepostProcessor: PrePostProcessor?
    
    public var imageView = UIImageView(frame: CGRect(x: 0, y: 0, width: 640, height: 640))
    
    public var resPointView = UIView(frame: CGRect(x: 0, y: 0, width: 160, height: 160))
    
    public var detectionImage: UIImage?
    public var finalImage: UIImage?
    
    public init(sku: String, ic: Int, initializeFinishNotice:(() -> Void)? = nil, finishFrameNotice: ((Bool) -> Void)? = nil ,frameNotice: ((DetectionEffectModel) -> Void)? = nil, doneNotice: ((DetectionResult?) -> Void)? = nil) {
        self.sku = sku
        self.ic = ic
        self.initializeFinishNotice = initializeFinishNotice
        self.finishFrameNotice = finishFrameNotice
        self.frameNotice = frameNotice
        self.doneNotice = doneNotice
        super.init()
        self.prepareConfig(sku: sku)
        self.setupBindings()
        self.startingIFrame()
        
    }
    
     func setupAV(finishHandler: ((Bool) -> Void)?) {
        // TODO: 进入后要先获取相机权限?
        AuthManager.instance.authorizedCameraWith { isSuccess in
            if !isSuccess {
                finishHandler?(false)
                //
                return
            }
        }
        // 初始化摄像头会话
        captureSession = AVCaptureSession()
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high
        // 获取当前摄像头输入 => 获取失败回调异常
        guard let videoDevice = AVCaptureDevice.default(for: .video) else { finishHandler?(false); return }
        guard let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else { finishHandler?(false); return }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        captureSession.commitConfiguration()
        // 获取当前帧图片
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue.main)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        // 注意AVFoundation start running需要在子线程
        DispatchQueue.global().async {
            self.captureSession.startRunning()
            // 设置预览图层 - 回调主线程
            DispatchQueue.main.async {
                self.previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
                finishHandler?(true)
            }
        }
    }
    
    func startCaptureFrame(step: Int) {
        print("log.f ==== 取第\(step)帧效果")
        let colorTwoDimArray = GHOpenCVBridge.shareManager().getColorsByStep(step)
        var greenArray: [Int] = []
        if colorTwoDimArray.count > 0 {
            greenArray = colorTwoDimArray[0].compactMap { $0.intValue }
        }
        var redArray: [Int] = []
        if colorTwoDimArray.count > 1 {
            redArray = colorTwoDimArray[1].compactMap { $0.intValue }
        }
        if greenArray.count > 0 && redArray.count > 0 {
            let effectModel = createEffectModel(step: step, greenArray: greenArray, redArray: redArray)
            self.frameNotice?(effectModel)
        }
    }
    
    // 启动摄像头
    func startingIFrame() {
        
        // 初始化识别侧灯效数组
        GHOpenCVBridge.shareManager().createAllStep(byIc: self.ic)
        
        self.setupAV { [weak self] isSuccess in
            guard let `self` = self else { return }
            if !isSuccess {
                // 回调异常结束 清空transaction
                self.transaction = nil
                self.doneNotice?(nil)
            } else {
                self.initializeFinishNotice?()
            }
        }
    }
    
    // 绑定接收回调
    func setupBindings() {
        // 开始回调
        self.startHandler = { [weak self] sessionId in
            guard let `self` = self else { return }
            if let oldSession = self.transaction {
                // 旧的流程还未结束 本次丢弃
            } else {
                // 新的流程 可以执行
                self.transaction = sessionId
                self.preImageArray.removeAll()
                self.afterImgArray.removeAll()
                // 返回第一帧灯效
                self.startCaptureFrame(step: 0)
            }
        }
        // 业务侧异常中断
        self.interruptHandler = { [weak self] in
            guard let `self` = self else { return }
            self.transaction = nil // 直接中断当前流程
        }
        // 取帧handler
        self.frameHandler = { [weak self] step in
            guard let `self` = self else { return }
            print("log.f ==== 收到第\(step)帧效果发送成功 取帧")
            if let _ = self.transaction { //正常transaction
                var second = 0.5
                if step == 0 {
                    second = 2 // 第一帧延时取
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + second) {
                    self.captureOneFrame()
                }
            }
        }
        
        // 取帧结束
        self.capFinishHandler = { [weak self] in
            guard let `self` = self else { return }
            // 对齐一帧
            if self.preImageArray.count > 0 {
                if self.preImageArray.count == GHOpenCVBridge.shareManager().getMaxStep() {
                    self.finishFrameNotice?(true)
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        self.alignmentAll { [weak self] in
                            guard let `self` = self else { return }
                            self.imageView.image = self.afterImgArray.first
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                                self.runDetection()
                            }
                        }
                    }
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        self.startCaptureFrame(step: self.preImageArray.count)
                    }
                }
            }
        }
    }
    
    // 截取一帧
    public func captureOneFrame() {
        self.needGetFrame = true
    }
    
    public func createEffectModel(step: Int, greenArray: [Int], redArray: [Int]) -> DetectionEffectModel {
        var colorDict: [UIColor: [Int]] = [:]
        colorDict[UIColor.red] = redArray
        colorDict[UIColor.green] = greenArray
        let effect = DetectionEffectModel(frameStep: step, colorDict: colorDict)
        return effect
    }
    
    func getImageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> UIImage? {
        let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        CVPixelBufferLockBaseAddress(imageBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelBuffer: CVPixelBuffer = imageBuffer!
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let image = UIImage(ciImage: ciImage)
        CVPixelBufferUnlockBaseAddress(imageBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        return image
    }

    // 此代理为主线程回调
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // 实时帧回调 这里只根据 业务侧closure回调取当前帧图像
        if self.needGetFrame {
            print("\n log.f ============= 开始取一帧")
            if let image = self.getImageFromSampleBuffer(sampleBuffer: sampleBuffer),let scaleImage = scaleImage(image, toSize: CGSize(width: 960, height: 1280)) {
                self.preImageArray.append(scaleImage)
                #if DEBUG
                self.imageView.image = scaleImage
                self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
                #endif
                DispatchQueue.main.asyncAfter(deadline: .now()+0.2) {
                    self.capFinishHandler?()
                }
            }
            self.needGetFrame = false
        }
    }
    
    func scaleImage(_ image: UIImage, toSize newSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
        image.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    
}

// MARK: configuration
extension GHDetectionTool {
    func prepareConfig(sku: String) {
        var cf = ProcessorConfig()
        if sku.hasPrefix("H682") {
            cf = ProcessorConfig(outputColumn: 6)
            switch sku {
            case "H6820":
                ic = 10
            case "H6821":
                ic = 20
            default: break
            }
            self.bizType = 1
        }
        prepostProcessor = PrePostProcessor(config: cf)
        ObjectDetector.instance.sku = sku
    }
}

// MARK: local detection
extension GHDetectionTool {
    
    func saveImageViewWithSubviewsToPhotoAlbum(imageView: UIImageView) {
        // 开始图形上下文
        UIGraphicsBeginImageContextWithOptions(imageView.bounds.size, false, 0.0)
        // 将imageView的layer渲染到图形上下文中
        imageView.layer.render(in: UIGraphicsGetCurrentContext()!)
        // 获取合成的图片
        let image = UIGraphicsGetImageFromCurrentImageContext()
        // 结束图形上下文
        UIGraphicsEndImageContext()
        // 保存图片到相册
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: image!)
        }, completionHandler: { success, error in
            if success {
                // 成功保存图片的操作
                print("图片已保存到相册")
            } else if let error = error {
                // 保存失败的操作
                print("保存图片出错: \(error.localizedDescription)")
            }
        })
    }
    
    func alignmentAll(finishHandler: () -> Void) {
        for (index, image) in self.preImageArray.enumerated() {
            let resImage = GHOpenCVBridge.shareManager().alignment(with:image, step: index, rotation: true)
            #if DEBUG
            self.imageView.image = resImage
            self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
            #endif
            self.afterImgArray.append(resImage)
        }
        
        if self.preImageArray.count == self.afterImgArray.count {
            finishHandler()
        }
    }
    
    func doneDetection(points: LightQueueBase) -> DetectionResult? {
        var pointsDict: [Int: [CGFloat]] = [:]
        for res in points.lightPoints {
            let ptArray = [CGFloat(res.x), CGFloat(res.y*3)]
            pointsDict[res.index] = ptArray
        }
        var anchorPoints: [[CGFloat]] = []
        // 直接拿四个点 变更点位置
        if !points.trapezoidalPoints.isEmpty {
            let res = points.trapezoidalPoints.removeLast()
            points.trapezoidalPoints.insert(res, at: 0)
            points.trapezoidalPoints.swapAt(2, 3) // 交换下标 1,2
        }
        for pt in points.trapezoidalPoints{
            anchorPoints.append([CGFloat(pt.x), CGFloat(pt.y*3)])
        }
        let result = DetectionResult(points: pointsDict, anchorPoints: anchorPoints, pixelScale: [960.0, 1280.0])
        return result
    }
    
    func runDetection() {
        // 只对第二张图进行识别
        if let prepostProcessor = self.prepostProcessor {
            let image = self.afterImgArray[0]
            let imageView = self.imageView
            self.imageView.image = image
            let imgScaleX = Double(image.size.width / CGFloat(prepostProcessor.inputWidth));
            let imgScaleY = Double(image.size.height / CGFloat(prepostProcessor.inputHeight));
            let ivScaleX : Double = (image.size.width > image.size.height ? Double(imageView.frame.size.width / image.size.width) : Double(imageView.frame.size.height / image.size.height))
            let ivScaleY : Double = (image.size.height > image.size.width ? Double(imageView.frame.size.height / image.size.height) : Double(imageView.frame.size.width / image.size.width))
            let startX = Double((imageView.frame.size.width - CGFloat(ivScaleX) * image.size.width)/2)
            let startY = Double((imageView.frame.size.height -  CGFloat(ivScaleY) * image.size.height)/2)
            guard var pixelBuffer = image.normalized() else {
                return
            }
            
            DispatchQueue.global().async {
                guard let outputs = self.inferencer.module.detect(image: &pixelBuffer) else {
                    return
                }
                let nmsPredictions = prepostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY)
                DispatchQueue.main.async {
                    prepostProcessor.showDetection(imageView: self.imageView, nmsPredictions: nmsPredictions, classes: self.inferencer.classes)
                    self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
                    var poArr: [PredictObject] = []
                    var ct = 0
                    for pre in nmsPredictions.filter({ $0.score > 0.2 && $0.classIndex != 2 }) {
                        ct+=1
                        let po = PredictObject()
                        po.type = pre.classIndex
                        po.x = pre.x
                        po.y = pre.y
                        po.width = pre.w
                        po.height = pre.h
                        po.score = CGFloat(pre.score)
                        po.lightId = ct
                        poArr.append(po)
                    }
                    
                    if poArr.count < 5 { // 少于5个 直接认为失败
                        self.doneNotice?(nil)
                        return
                    }
                    
                    GHOpenCVBridge.shareManager().clearAllresLp()
                    GHOpenCVBridge.shareManager().createLightPointArray(poArr)
                    
                    var resultJsonString = ""
                    for (idx, _) in self.afterImgArray.enumerated() {
                        let jsonStr =  GHOpenCVBridge.shareManager().caculateNum(byStep: idx, bizType: self.bizType)
                        if idx == self.afterImgArray.count-1 {
                            resultJsonString = jsonStr
                        }
                    }
                    print("log.f result json string \(resultJsonString)")
                    
                    if let data = resultJsonString.data(using: .utf8) {
                        let dt = try?JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                        let pointbase = LightQueueBase.deserialize(from: dt)
                        if let pt = pointbase, !pt.lightPoints.isEmpty && !pt.trapezoidalPoints.isEmpty {
                            let detectionResult = self.doneDetection(points: pt)
                            self.doneNotice?(detectionResult)
                            let image = GHOpenCVBridge.shareManager().showLastOutlet()
                            self.finalImage = image
                            self.imageView.image = image
                            #if DEBUG
                            self.resPointView.backgroundColor = UIColor.white
                            for basRes in pt.lightPoints {
                                // frame需要转换！！！
                                let rectView = UIView(frame: CGRect(x: basRes.x/12, y: basRes.y/8, width: 2, height: 2))
                                rectView.backgroundColor = UIColor.green
                                self.resPointView.addSubview(rectView)
                            }
                            for traRes in pt.trapezoidalPoints {
                                // frame需要转换！！！
                                let rectView = UIView(frame: CGRect(x: traRes.x/12, y: traRes.y/8, width: 2, height: 2))
                                rectView.backgroundColor = UIColor.blue
                                self.resPointView.addSubview(rectView)
                            }
                            self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
                            #endif
                        } else {
                            self.doneNotice?(nil)
                        }
                    }
                }
            }
        }
    }
}
