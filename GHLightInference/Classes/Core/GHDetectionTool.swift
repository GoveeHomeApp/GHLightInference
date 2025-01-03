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
    /* ========== 通知业务层closure ========== */
    // 初始化完成Notice
    public var initializeFinishNotice: (() -> Void)?
    // 取帧流程结束Notice
    public var finishFrameNotice: ((Bool) -> Void)?
    // 取帧Notice（每一帧需要的灯效， 开始成功就返回第一帧）
    public var frameNotice: ((DetectionEffectModel) -> Void)?
    // 完成Notice
    public var doneNotice: ((DetectionResult?) -> Void)?
    
    // 识别流程外 结果Notice（注意此closure不可以出现在识别流程中）
    public var showCurrentNotice: ((DetectionResult?) -> Void)?
    
    /* ========== 业务层回调closure ========== */
    // 开始 & 重新开始 带有transaction
    public private(set) var startHandler: ((String) -> Void)?
    // 中断
    public private(set) var interruptHandler: (() -> Void)?
    // 取帧（第几步 ps:第几次取帧 从序号1开始）
    public private(set) var frameHandler:((Int) -> Void)?
    // AVCaptureDevice 相关参数调整
    public private(set) var exposureTargetBiasHandler:((Double) -> Void)?
    // 识别流程外当前调整亮度或者对比度下展示识别第一帧结果（注意此closure不可以出现在识别流程中）
    public private(set) var showCurrentHandler:((_ layerRect: CGRect, _ backView: UIView) -> Void)?
    
    // 业务层结束了 停止轮询识别
    public private(set) var stopReconHandler:(() -> Void)?
    
    
    
    /* ========== 私有closure ========== */
    // 是否为识别流程中的cap
    private var capFinishHandler: (() -> Void)?
    // 是否在当前识别流程内 => 通过transaction去判断
    // 间隔3s识别一次
    let queue: DispatchQueue = DispatchQueue.main
    let group = DispatchGroup()
    
    private var realFrame: CGRect?
    private var backView: UIView?
    private var rectView = UIView()
    private var preImg: UIImage?
    
    private var isHori: Bool = false
    private var minus: CGFloat = 0.0
    
    // 识别流程唯一transaction 每次整体流程都是唯一的 结束会被置空
    private var transaction: String? {
        didSet {
            if let t = transaction {
                
            } else {
                self.preImageArray.removeAll()
                self.afterImgArray.removeAll()
            }
        }
    }
    // 业务参数
    public var sku: String = ""
    public var ic: Int = 0
    public var dimension: String = "3D"
    // 0 - H70CX 1 - H682X
    var bizType: Int = 0
    
    // 取帧相关
    var captureSession: AVCaptureSession!
    var videoOutput: AVCaptureVideoDataOutput!
    var captureDevice: AVCaptureDevice!
    // 业务层直接拿的展示Layer
    public private(set) var previewLayer: AVCaptureVideoPreviewLayer?
    // 取帧参数
    var needGetFrame: Bool = false
    // 对齐前后图片数组
    var preImageArray: [UIImage] = []
    var afterImgArray: [UIImage] = []
    // 识别流程对象
    private var inferencer = GHLightDetectManager.instance.detector()
    private var prepostProcessor: PrePostProcessor?
    // DEBUG专用
    public var imageView = UIImageView(frame: CGRect(x: 0, y: 0, width: 640, height: 640))
    public var saveImageView = UIImageView(frame: CGRect(x: 0, y: 0, width: 1280, height: 960))
    public var resPointView = UIView(frame: CGRect(x: 0, y: 0, width: 160, height: 160))
    public var detectionImage: UIImage?
    public var finalImage: UIImage?
    private var sc:(Double, Double) = (0.0, 0.0)
    
    public init(sku: String, ic: Int, dimension: String, initializeFinishNotice:(() -> Void)? = nil, finishFrameNotice: ((Bool) -> Void)? = nil ,frameNotice: ((DetectionEffectModel) -> Void)? = nil, doneNotice: ((DetectionResult?) -> Void)? = nil) {
        self.sku = sku
        self.ic = ic
        self.dimension = dimension
        self.initializeFinishNotice = initializeFinishNotice
        self.finishFrameNotice = finishFrameNotice
        self.frameNotice = frameNotice
        self.doneNotice = doneNotice
        super.init()
        // 准备识别参数
        self.prepareConfig(sku: sku)
        // 初始化回调
        self.setupBindings()
        // 启动AVFoundation
        self.startingIFrame()
    }
    
    // 初始化AVCaptureSession
    func setupAV(finishHandler: ((Bool) -> Void)?) {
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
        
        self.captureDevice = videoDevice
        
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
    
    // 开始第X帧截取 通知业务层灯效
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
        
        self.stopReconHandler = { [weak self] in
            guard let `self` = self else { return }
            
        }
        
        // 识别一次 就会调用这个
        self.showCurrentHandler = { [weak self] (cz, v) in
            guard let `self` = self else { return }
            var oX = 0.0
            if let oriR = self.realFrame {
                oX = oriR.origin.y
                print("log.p ====== oX \(oX)")
            }
            self.rectView.isHidden = false
            self.backView = v
            self.rectView.removeFromSuperview()
            self.realFrame = cz
            self.rectView.frame = cz
            self.isHori = rectView.frame.size.width > rectView.frame.size.height
            if self.isHori {
                if self.minus <= 0 && oX > 0 {
                    self.minus = oX - cz.origin.x
                }
            } else {
                self.minus = 0.0
            }
            self.rectView.backgroundColor = UIColor.clear
            self.backView?.addSubview(rectView)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.captureOneFrame()
            }
        }
        
        // -1最暗 1最亮
        self.exposureTargetBiasHandler = { [weak self] expBias in
            guard let `self` = self else { return }
            if self.captureDevice.isExposureModeSupported(.autoExpose) && self.captureDevice.isExposurePointOfInterestSupported {
                // 设备支持亮度调节
                do {
                    try self.captureDevice.lockForConfiguration()
                    self.captureDevice.setExposureTargetBias(Float(expBias), completionHandler: nil) // 调整亮度到中间值
                    self.captureDevice.unlockForConfiguration()
                } catch {
                    print("无法锁定设备配置以进行亮度调节")
                }
            } else {
                print("此设备不支持亮度调节")
            }
        }
        
        // 开始回调
        self.startHandler = { [weak self] sessionId in
            guard let `self` = self else { return }
            if let oldSession = self.transaction {
                // 旧的流程还未结束 本次丢弃
            } else {
                // 新的流程 可以执行
                self.rectView.isHidden = true
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
                var second = 0.1
                if step == 0 {
                    second = 0.3 // 第一帧延时取
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
                        // 先调用识别
                        self.runBatchDetect { [weak self] success in
                            guard let `self` = self else { return }
                            print("log.ppp ==== \(success)")
                            if success {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                    self.alignmentAll { [weak self] in
                                        guard let `self` = self else { return }
                                        self.imageView.image = self.afterImgArray.first
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                            self.runDetection()
                                        }
                                    }
                                }
                            } else {
                                // 个数太少直接抛失败
                                self.doneFailed()
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
    
    // 处理数组
    func addNumbers(start: Int, end: Int, to array: inout [Int]) {
        for number in start...end {
            array.append(number)
        }
    }
    
    // 灯效组装
    public func createEffectModel(step: Int, greenArray: [Int], redArray: [Int]) -> DetectionEffectModel {
        var redArr = redArray
        var greenArr = greenArray
        var colorDict: [UIColor: [Int]] = [:]
        switch self.bizType {
        case 2:
            redArr = []
            greenArr = []
            // H6820/1 处理颜色数组
            for greenLamp in greenArray {
                if greenLamp == 0 {
                    addNumbers(start: 0, end: 10, to: &greenArr)
                }  else {
                    addNumbers(start: greenLamp*11, end: (greenLamp + 1)*11 - 1, to: &greenArr)
                }
            }
            for redLamp in redArray {
                if redLamp == 0 {
                    addNumbers(start: 0, end: 10, to: &redArr)
                }  else {
                    addNumbers(start: redLamp*11, end: (redLamp + 1)*11 - 1, to: &redArr)
                }
            }
        default:
            break
        }
        colorDict[UIColor.red] = redArr
        colorDict[UIColor.green] = greenArr
        let effect = DetectionEffectModel(frameStep: step, colorDict: colorDict)
        return effect
    }
    
    // 取帧图像处理
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
            if let _ = self.transaction {
                print("\n log.f ============= 开始取一帧")
                if let image = self.getImageFromSampleBuffer(sampleBuffer: sampleBuffer), let scaleImage = scaleImage(image, toSize: CGSize(width: 960, height: 1280)) {
                    print("log.f ====== width\(image.size.width) height\(image.size.height)")
                    self.sc = (image.size.width, image.size.height)
                    self.preImageArray.append(scaleImage)
                    #if DEBUG
                    self.saveImageView.image = scaleImage
                    //                self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.saveImageView)
                    #endif
                    DispatchQueue.main.asyncAfter(deadline: .now()+0.1) {
                        self.capFinishHandler?()
                    }
                }
                self.needGetFrame = false
            } else {
                // 只识别 不跑业务
                if let preLayer = self.previewLayer, let image = self.getImageFromSampleBuffer(sampleBuffer: sampleBuffer) {
                    if let fra = self.realFrame {
                        self.saveImageView.image = image
//                        self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.saveImageView)
                        print("log.f ====== width \(fra.width) height \(fra.height)")
                        if let rotat = image.rotated(by: .pi/2) {
//                            self.saveImageToAlbum(image: rotat)
                            print("log.f ======  rotat image width \(rotat.size.width) rotat image height \(rotat.size.height)")
                            self.preImg = rotat
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                self.runDetection(false)
                            }
                        }
                    }
                }
                self.needGetFrame = false
            }
        }
    }
    // 牺牲质量放缩
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
    
    func doneFailed() {
        DispatchQueue.main.async {
            // 抛出识别失败
            self.doneNotice?(nil)
            self.transaction = nil
        }
    }
    
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
            self.bizType = 2
        } else if sku.hasPrefix("H70C") || sku.hasPrefix("H80C") {
            self.bizType = self.dimension == "3D" ? 0 : 1
        }
        print("log.f ====== dimension \(self.dimension)")
        prepostProcessor = GHLightDetectManager.instance.prepostProcessor(config: cf)
        self.inferencer.sku = sku
        self.inferencer.dimension = self.dimension
    }
}

// MARK: local detection
extension GHDetectionTool {
    
    func saveImageToAlbum(image: UIImage) {
        PHPhotoLibrary.requestAuthorization { status in
            if status == .authorized {
                PHPhotoLibrary.shared().performChanges({
                    let request = PHAssetChangeRequest.creationRequestForAsset(from: image)
                    request.creationDate = Date()
                }) { success, error in
                    if success {
                        print("图片已成功保存到相册")
                    } else {
                        print("保存图片失败: \(error?.localizedDescription ?? "未知错误")")
                    }
                }
            } else {
                print("用户未授权访问相册")
            }
        }
    }
    
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
    // 对齐
    func alignmentAll(finishHandler: () -> Void) {
        do  {
            for (index, image) in self.preImageArray.enumerated() {
                // 对齐添加多线程
                let resImage = GHOpenCVBridge.shareManager().alignment(with:image, step: index, rotation: true) { [weak self] err in
                    self?.doneFailed()
                    return
                }
                #if DEBUG
                self.imageView.image = resImage
                #endif
                self.afterImgArray.append(resImage)
            }
            
            if self.preImageArray.count == self.afterImgArray.count {
                finishHandler()
            }
        } catch {
            print("log.f ====== RFAILED Caught a Swift error: \(error)")
            self.doneFailed()
        }
    }
    // 结果组装返回
    func doneDetection(points: LightQueueBase) -> DetectionResult? {
        // H6820 不需要放缩
        let scaleY:CGFloat = self.bizType != 2 ? sc.0 : 1
        let scaleX:CGFloat = self.bizType != 2 ? sc.1 : 1
        
        var pointsDict: [Int: [CGFloat]] = [:]
        var anchorPoints: [[CGFloat]] = []
        for res in points.lightPoints {
            let ptArray = [CGFloat(res.x)*scaleX, CGFloat(res.y)*scaleY]
            pointsDict[res.index] = ptArray
        }
        // 直接拿四个点 变更点位置
        if !points.trapezoidalPoints.isEmpty {
            var xr = 0
            var xl = 0
            for p in points.trapezoidalPoints {
                // 左上点
                if p.pName == "lT" {
                    xl = p.x
                }
                if p.pName == "rT" {
                    xr = p.x
                }
            }
            let width = CGFloat((xr-xl)/2) * 0.2
            print("log.f ====== real half width\(width)")
            for p in points.trapezoidalPoints {
                if p.pName == "lT" {
                    p.x = Int(CGFloat(p.x)+CGFloat(width))
                }
                if p.pName == "rT" {
                    p.x = Int(CGFloat(p.x)-CGFloat(width))
                }
            }
            
            let res = points.trapezoidalPoints.removeLast()
            points.trapezoidalPoints.insert(res, at: 0)
            points.trapezoidalPoints.swapAt(2, 3) // 交换下标 1,2
            for pt in points.trapezoidalPoints{
                anchorPoints.append([CGFloat(pt.x)*scaleX, CGFloat(pt.y)*scaleY])
            }
        }
        let result = DetectionResult(points: pointsDict, anchorPoints: anchorPoints, pixelScale: [960.0, 1280.0], objectPoints: points.lightPoints, preImageArray: self.preImageArray)
        return result
    }
    
    func runBatchDetect(complete: @escaping(Bool) -> Void) {
        if self.preImageArray.isEmpty { complete(false) }
        var resArr: [Int] = []
        for (idx, img) in self.preImageArray.enumerated() {
            group.enter()
            queue.async(group: group) {
                self.runOnlyDetect(pre: img) { [weak self] ct in
                    resArr.append(ct)
                    print("log.ppp ====== \(resArr)")
                    self?.group.leave()
                }
            }
        }
        group.notify(queue: .main) {
            let fin = resArr.filter { $0 < 20 }
            complete(fin.isEmpty)
        }
    }
    
    // 识别灯珠onlyDetect
    func runOnlyDetect(pre: UIImage, complete: @escaping(Int) -> Void) {
        if let oriimage = scaleImage(pre, toSize: CGSize(width: 640, height: 640)), let prepostProcessor = self.prepostProcessor {
            // 必须得有这一步 不然就会有问题
            let image = GHOpenCVBridge.shareManager().alignment(with:oriimage, step: 0, rotation: false) { [weak self] err in complete(0); return }
            let imgScaleX = Double(pre.size.width / CGFloat(prepostProcessor.inputWidth));
            let imgScaleY = Double(pre.size.height / CGFloat(prepostProcessor.inputHeight));
            let ivScaleX : Double = Double(rectView.frame.size.width / pre.size.width)
            let ivScaleY : Double = Double(rectView.frame.size.width / pre.size.width)
            let startX = Double((rectView.frame.size.width - CGFloat(ivScaleX) * pre.size.width)/2)
            let startY = Double((rectView.frame.size.height -  CGFloat(ivScaleY) * pre.size.height)/2)
            guard var pixelBuffer = image.normalized() else { complete(0); return }
            DispatchQueue.global().async {
                var outputs: [NSNumber] = []
                // 处理识别异常捕获
                if let op = self.inferencer.module.detect(image: &pixelBuffer) {
                    outputs = op
                    // 预测数据
                    let nmsPredictions = prepostProcessor.originOutputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY)
                    let res = nmsPredictions.filter { $0.score > 0.2 }
                    complete(res.count)
                } else {
                    complete(0)
                }
            }
        }
    }
    
    // 识别灯珠流程
    func runDetection(_ isInProcess: Bool = true) {
        // 只对第一张图进行识别
        if let prepostProcessor = self.prepostProcessor, !self.afterImgArray.isEmpty {
            let image = self.afterImgArray[0]
            let imageView = self.imageView
            self.imageView.image = image
            print("log.p ====== image w: \(image.size.width) h: \(image.size.height)")
            print("log.p ====== w: \(imageView.frame.size.width) h: \(imageView.frame.size.height)")
//            self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
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
                var outputs: [NSNumber] = []
                // 处理识别异常捕获
                do  {
                    if let op = self.inferencer.module.detect(image: &pixelBuffer) {
                        outputs = op
                    } else {
                        self.doneFailed()
                    }
                } catch let error as NSError {
                    print("log.f ====== RFAILED Caught an Objective-C exception: \(error.localizedDescription)")
                    self.doneFailed()
                } catch {
                    print("log.f ====== RFAILED Caught a Swift error: \(error)")
                    self.doneFailed()
                }
                // 预测数据
                let nmsPredictions = prepostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY)
                
                // 回调主线程绘图
                DispatchQueue.main.async {
                    prepostProcessor.showDetection(imageView: self.imageView, nmsPredictions: nmsPredictions, classes: self.inferencer.classes)
                    #if DEBUG
//                    self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
                    #endif
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
                    print("log.f ======== \(poArr.count)")
                    GHOpenCVBridge.shareManager().clearAllresLp()
                    switch self.bizType {
                    case 2:
                        // 6820系列
                        if poArr.count < 1 { // 少于5个 直接认为失败
                            self.doneNotice?(nil)
                            self.transaction = nil
                            return
                        }
                        GHOpenCVBridge.shareManager().createLightPointArray(poArr, withBiz: self.bizType)
                        break
                    default:
                        if poArr.count < 5 { // 少于5个 直接认为失败
                            self.doneNotice?(nil)
                            self.transaction = nil
                            return
                        }
                        GHOpenCVBridge.shareManager().createLightPointArray(poArr, withBiz: self.bizType)
                    }
                    
                    var resultJsonString = ""
                    for (idx, _) in self.afterImgArray.enumerated() {
                        let jsonStr =  GHOpenCVBridge.shareManager().caculateNum(byStep: idx, bizType: self.bizType) { [weak self] err in
                            self?.doneFailed()
                            return
                        }
                        if idx == self.afterImgArray.count-1 {
                            resultJsonString = jsonStr
                        }
                    }
                    print("log.f result json string \(resultJsonString)")
                    if let data = resultJsonString.data(using: .utf8) {
                        let dt = try?JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                        let pointbase = LightQueueBase.deserialize(from: dt)
                        if let pt = pointbase, !pt.lightPoints.isEmpty && pt.lightPoints.count > 10 {
                            var indexArr: [Int] = []
                            if self.bizType == 0 {
                                print("Count \(pt.lightPoints.count)")
                                let ptArr = pt.lightPoints.sorted { $0.index < $1.index }
                                for i in 0 ... self.ic/10 {
                                    // 每十个点检查一组，将一组内y值偏差较大的值剔除
                                    let range: (Int, Int) = (i*10, (i+1)*10 - 1)
                                    let arr = ptArr.filter { $0.index >= range.0 && $0.index <= range.1 }.sorted { $0.y < $1.y }
                                    // 不大于二 就不管了
                                    if arr.count > 2 {
                                        var temp = arr
                                        temp.remove(at: 0)
                                        temp.removeLast()
                                        let average = temp.map { $0.y }.reduce(0) { $0 + $1 }/temp.count
                                        for pt in arr {
                                            let val = abs(pt.y - average)
                                            if pt.y > 400 {
                                                if val > 50 {
                                                    indexArr.append(pt.index)
                                                }
                                            } else {
                                                if val > 30 {
                                                    indexArr.append(pt.index)
                                                }
                                            }
                                        }
                                    } else if arr.count == 2 {
                                        let first = arr[0]
                                        let last = arr[1]
                                        if abs(first.y-last.y) > 30 {
                                            // 两个点差的太远了
                                            // 找出比两个点序号小的最大序号点
                                            let res = ptArr.filter { $0.index < first.index && $0.index < last.index }.sorted { $0.index < $1.index }.last
                                            if let r = res {
                                                let v1 = abs(r.y - first.y)
                                                let v2 = abs(r.y - last.y)
                                                if v1 > 30 && v2 <= 30 {
                                                    indexArr.append(first.index)
                                                } else if v1 <= 30 && v2 > 30 {
                                                    indexArr.append(last.index)
                                                } else {
                                                    indexArr.append(first.index)
                                                    indexArr.append(last.index)
                                                }
                                            }
                                        }
                                    }
                                }
                                print("log.f ===== \(indexArr.count)")
                                let _ = indexArr.map { print("log.f ==== index is \($0)") }
                                if !indexArr.isEmpty {
                                    for pp in ptArr {
                                        if indexArr.contains(pp.index) {
                                            pp.isBad = true
                                        }
                                    }
                                }
                                print("log.f ===== pre ct \(pt.lightPoints.count)")
                                pt.lightPoints = ptArr.filter { !$0.isBad }
                                print("log.f ===== aft ct \(pt.lightPoints.count)")
                            }
                            
                            let detectionResult = self.doneDetection(points: pt)
                            self.doneNotice?(detectionResult)
                            self.transaction = nil
                        } else {
                            self.doneFailed()
                        }
                    }
                }
            }
        } else {
            if let pre = self.preImg, let prepostProcessor = self.prepostProcessor, !isInProcess {
                
                if let oriimage = scaleImage(pre, toSize: CGSize(width: 640, height: 640)) {
                    // 必须得有这一步 不然就会有问题
                    let image = GHOpenCVBridge.shareManager().alignment(with:oriimage, step: 0, rotation: false) { [weak self] err in }
//                    self.saveImageToAlbum(image: oriimage)
                    let imageView = self.imageView
                    self.imageView.image = image
                    
                    let imgScaleX = Double(pre.size.width / CGFloat(prepostProcessor.inputWidth));
                    let imgScaleY = Double(pre.size.height / CGFloat(prepostProcessor.inputHeight));
                    
                    let ivScaleX : Double = Double(rectView.frame.size.width / pre.size.width)
                    let ivScaleY : Double = Double(rectView.frame.size.width / pre.size.width)

                    let startX = Double((rectView.frame.size.width - CGFloat(ivScaleX) * pre.size.width)/2)
                    let startY = Double((rectView.frame.size.height -  CGFloat(ivScaleY) * pre.size.height)/2)
                    
                    guard var pixelBuffer = image.normalized() else {
                        return
                    }
                    
                    DispatchQueue.global().async {
                        var outputs: [NSNumber] = []
                        // 处理识别异常捕获
                        if let op = self.inferencer.module.detect(image: &pixelBuffer) {
                            outputs = op
                            // 预测数据
                            let nmsPredictions = prepostProcessor.originOutputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY)
                            // 回调主线程绘图
                            DispatchQueue.main.async {
                                print("log.p ======= count \(nmsPredictions.count)")
                                prepostProcessor.showPreDetection(view: self.rectView, nmsPredictions: nmsPredictions, classes: self.inferencer.classes, self.isHori, self.minus)
                                // 回调识别结果 组装结果业务数据
                                var res = DetectionResult()
                                var finalnms = nmsPredictions.filter { $0.score >= 0.2 }
                                for (ind, eachPredict) in finalnms.enumerated() {
                                    res.points[ind] = [CGFloat(eachPredict.x), CGFloat(eachPredict.y)]
                                }
                                self.showCurrentNotice?(res)
                                GHOpenCVBridge.shareManager().releaseOutProcess()
                                #if DEBUG
//                                self.saveImageViewWithSubviewsToPhotoAlbum(imageView: self.imageView)
                                #endif
                            }
                        }
                    }
                }
            } else {
                self.doneFailed()
            }
        }
    }
}
