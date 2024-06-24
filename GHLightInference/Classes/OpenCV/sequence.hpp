#ifndef __SEQUENCE_HPP__
#define __SEQUENCE_HPP__

#include "common.hpp"
#include "features.hpp"

using namespace cv;
using namespace std;

struct PointIndex {
    vector<int> list;
    int score;

    PointIndex(vector<int> l = {}, int s = 0) : list(l), score(s) {}
};

/**
 * 初始化构建逻辑
 */
int initVector(int icNum);

/**
 * 获取当前颜色多重数组
 * @param frameStep
 * @param icNum
 * @return
 */
vector<vector<int>> getColors(int frameStep);

/**
 * 最大取帧轮数
 */
int getMaxStepCnt();

/**
 * 最小得分
 */
int getScoreMin();

int getIcNum();

int checkIsGreen(int index);

/**
 * 最大得分
 */
int getScoreMax();

/**
 * 灯光同色场景
 */
vector<int> getSameColorVector();

/**
 * 根据得分计算灯序号
 */
void setLightIndex(LightPoint &point, int score, int lightType);

int getLightIndex(int score, int lightType);

#endif
