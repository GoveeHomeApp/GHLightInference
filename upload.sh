#!/bin/bash

Cyan='\033[0;36m'
Default='\033[0;m'

moduleName=$1
tagString=$2
specFilePath=$3
NewVersionNumber=""

#修改spec文件tag
#获取版本号并显示
VersionString=`grep -E "s.version.*=" GHLightInference.podspec`
VersionNumberDot=`tr -cd "[0-9.]" <<<"$VersionString"`
VersionNumber=`sed 's/^.//' <<<"$VersionNumberDot"`
NewVersionNumber=$tagString

echo -e "\n${Default}================================================"
echo -e " Current Version   :  ${Cyan}${VersionNumber}${Default}"
echo -e "================================================\n"

#输出当前版本号
echo -e "\n${Default}================================================"
echo -e " New Version IS :  ${Cyan}${NewVersionNumber}${Default}"
echo -e "================================================\n"

LineNumber=`grep -nE 's.version.*=' GHLightInference.podspec | cut -d : -f1`
sed -i "" "${LineNumber}s/${VersionNumber}/${NewVersionNumber}/g" GHLightInference.podspec
echo -e "\n${Default}================================================"
echo -e "current version is ${VersionNumber}, new version is ${NewVersionNumber}"
echo -e "================================================\n"
