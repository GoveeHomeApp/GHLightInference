//
//  AuthManager.swift
//  GHLightDetection
//
//  Created by sy on 2024/5/16.
//

import Foundation
import AVFoundation
import Photos
import AssetsLibrary

public class AuthManager {
    
    public private(set) static var instance = AuthManager()
    
    /// 获取相册权限
    public func authorizedPhotoWith(completion: @escaping (Bool) -> Void) {
        let granted = PHPhotoLibrary.authorizationStatus()
        switch granted {
        case PHAuthorizationStatus.authorized:
            completion(true)
        case PHAuthorizationStatus.denied, PHAuthorizationStatus.restricted:
            completion(false)
        case PHAuthorizationStatus.notDetermined:
            PHPhotoLibrary.requestAuthorization({ status in
                completion(status == PHAuthorizationStatus.authorized)
            })
        case .limited:
            completion(true)
        @unknown default:
            completion(false)
        }
    }

    /// 相机权限
    public func authorizedCameraWith(completion: @escaping (Bool) -> Void) {
        let granted = AVCaptureDevice.authorizationStatus(for: AVMediaType.video)
        switch granted {
        case .authorized:
            completion(true)
        case .denied:
            completion(false)
        case .restricted:
            completion(false)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: AVMediaType.video, completionHandler: { (granted: Bool) in
                completion(granted)
            })
        @unknown default:
            completion(false)
        }
    }
}


