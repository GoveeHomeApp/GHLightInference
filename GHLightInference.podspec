Pod::Spec.new do |s|

  s.name         = 'GHLightInference'
  s.version      = '0.0.0'
  s.summary      = 'GHLightInference.'

  s.homepage     = 'git@github.com:GoveeHomeApp/GHLightInference.git'

  s.license      = { :type => 'MIT', :file => 'LICENSE' }

  s.author       = 'sy'

  s.ios.deployment_target = '13.0'

  s.swift_version = '5.0'

  s.source       = { :git => 'git@github.com:GoveeHomeApp/GHLightInference.git', :tag => s.version.to_s }

  s.source_files = 'GHLightInference/Classes/**/*'
  
  s.private_header_files = 'GHLightInference/Classes/OpenCV/**/*.{hpp,h}'

  #- 推荐这种 -#
  s.resource_bundles = { 'GHLightInference' => ['GHLightInference/Assets/**/*.{xcassets,png,torchscript.ptl,json,plist,txt}'] }
  s.vendored_frameworks = ['opencv2.framework']
  s.xcconfig = { 'HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_ROOT)/LibTorch-Lite/install/include/"' }
  s.dependency 'LibTorch-Lite', '~>1.13.0.1'
  s.dependency 'HandyJSON'
#  s.dependency 'OpenCV2'
  
end
