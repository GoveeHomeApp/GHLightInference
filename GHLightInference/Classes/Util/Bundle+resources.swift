//
//  Bundle+resources.swift
//  GHLightDetection
//
//  Created by sy on 2024/5/17.
//

import Foundation

public extension Bundle {

    public static func bundleName(of bundleClass: AnyClass) -> URL? {
        // 以动态库导入的bundle
        if let bundleName = Bundle(for: bundleClass.self).infoDictionary?[kCFBundleNameKey as String] as? String,
         let bundleUrl = Bundle(for: bundleClass.self).url(forResource: bundleName, withExtension: "bundle") {
          return bundleUrl
        }

        // 以静态库导入的bundle
        if let bundleName = String.init(reflecting: bundleClass.self).split(separator: ".").first?.description,
          let bundleUrl = Bundle.main.url(forResource: bundleName, withExtension: "bundle") {
          return bundleUrl
        }
        
        return nil
    }
    
    public static func bundleResource(of bundleClass: AnyClass) -> Bundle? {
        guard let url = bundleName(of: bundleClass),
            let resourceBundle = Bundle(url: url)
            else { return nil }
        return resourceBundle
    }
    
    public static func resource(name: String, ofType: String? = nil, in bundleClass: AnyClass) -> Any? {
        guard let bundle = bundleResource(of: bundleClass) else { return nil }
        return bundle.path(forResource: name, ofType: ofType)
    }
}
