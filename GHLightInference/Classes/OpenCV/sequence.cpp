#include <iostream>
#include "sequence.hpp"

const int red = -65536;
const int green = -16711936;
const int blue = -16776961;
const vector<int> numbers = {red, green, blue};
vector <PointIndex> listResult;
int icNumber = 500;

/**
 * 初始化构建逻辑
 */
int initVector(int icNum) {
    if (icNumber == icNum && !listResult.empty()) {
        LOGW(LOG_TAG, "已经初始化过了 getMaxStepCnt()=%d", getMaxStepCnt());
        return getMaxStepCnt();
    }
    icNumber = icNum;
    LOGW(LOG_TAG, "initVector icNumber=%d", icNumber);
    int vector1[] = {1, 2};
    int vector2[] = {3, 5};
    int vector3[] = {6, 10};
    int vector4[] = {11, 19};//15
    int vector5[] = {20, 36};//31
    int vector6[] = {37, 69};//78
    int vector7[] = {70, 134};//140-78=62
    int vector8[] = {135, 263};//126
    int vector9[] = {264, 520};//531- 280=248
    int frameStep = 8;
    vector <PointIndex> listPointIndex;
    for (int index1 = 0; index1 < 2; index1++) {
        int s1 = vector1[index1];
        for (int index2 = 0; index2 < 2; index2++) {
            int s2 = vector2[index2];
            for (int index3 = 0; index3 < 2; index3++) {
                int s3 = vector3[index3];
                for (int index4 = 0; index4 < 2; index4++) {
                    int s4 = vector4[index4];
                    for (int index5 = 0; index5 < 2; index5++) {
                        int s5 = vector5[index5];
                        for (int index6 = 0; index6 < 2; index6++) {
                            int s6 = vector6[index6];
                            for (int index7 = 0; index7 < 2; index7++) {
                                int s7 = vector7[index7];
                                if (getMaxStepCnt() == 7) {
                                    frameStep = 7;
                                    PointIndex p = PointIndex();
                                    p.list = {index1, index2, index3, index4, index5, index6,
                                              index7};
                                    p.score = s1 + s2 + s3 + s4 + s5 + s6 + s7;
                                    if ((p.score != getScoreMin()) && (p.score != getScoreMax()) &&
                                        (p.score - getScoreMin() - 2) < icNum) {
                                        listPointIndex.push_back(p);
                                    }
                                } else {
                                    for (int index8 = 0; index8 < 2; index8++) {
                                        int s8 = vector8[index8];
                                        if (getMaxStepCnt() == 8) {
                                            frameStep = 8;
                                            PointIndex p = PointIndex();
                                            p.list = {index1, index2, index3, index4, index5,
                                                      index6,
                                                      index7, index8};
                                            p.score = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
                                            if ((p.score != getScoreMin()) &&
                                                (p.score != getScoreMax()) &&
                                                (p.score - getScoreMin() - 2) < icNum) {
                                                listPointIndex.push_back(p);
                                            }
                                        } else {
                                            for (int index9 = 0; index9 < 2; index9++) {
                                                int s9 = vector9[index9];
                                                PointIndex p = PointIndex();
                                                p.list = {index1, index2, index3, index4, index5,
                                                          index6, index7, index8, index9};
                                                p.score =
                                                        s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
                                                if ((p.score != getScoreMin()) &&
                                                    (p.score != getScoreMax()) &&
                                                    (p.score - getScoreMin() - 2) < icNum) {
                                                    listPointIndex.push_back(p);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    listResult = listPointIndex;
    sort(listResult.begin(), listResult.end(), [](const PointIndex &a, const PointIndex &b) {
        return a.score < b.score;
    });
    return frameStep;
}

/**
 * 获取当前颜色多重数组
 * @param frameStep
 * @param icNum
 * @return
 */
vector <vector<int>> getColors(int frameStep) {
    vector <vector<int>> colorMap;
    vector<int> indexList;

    for (const auto &p: listResult) {
        indexList.push_back(p.list[frameStep]);
    }
    LOGW(LOG_TAG, "getColors frameStep=%d  indexList=%d", frameStep, indexList.size());
    int cntOddIndex = 0;
    int cntEvenIndex = 0;
    for (int index = 0; index < icNumber; index++) {
        if (index % 2 == 0) {
            if (cntOddIndex > indexList.size()) {
                LOGE(LOG_TAG, "cntOddIndex > indexList,  cntOddIndex=%d,  index=%d", cntOddIndex,
                     index);
                continue;
            }
            int colorIndex = indexList[cntOddIndex];
            if (colorIndex > 2) {
                LOGE(LOG_TAG, "colorIndex > 2,  cntOddIndex=%d,  index=%d,  colorIndex=%d",
                     cntOddIndex,
                     index, colorIndex);
                continue;
            }
            colorMap.push_back({index, numbers[colorIndex]});
            cntOddIndex++;
        } else {
            colorMap.push_back({index, numbers[cntEvenIndex]});
            cntEvenIndex++;
            if (cntEvenIndex == 2) {
                cntEvenIndex = 0;
            }
        }
    }
    return colorMap;
}

/**
 * 最大取帧轮数
 */
int getMaxStepCnt() {
    if (icNumber < 255) {
        return 7;
    } else if (icNumber < 511) {
        return 8;
    } else {
        return 9;
    }
}

/**
 * 最小得分
 */
int getScoreMin() {
    if (getMaxStepCnt() == 7) {
        return 148;
    } else if (getMaxStepCnt() == 8) {
        return 283;
    } else {
        return 547;
    }
}

/**
 * 最大得分
 */
int getScoreMax() {
    if (getMaxStepCnt() == 7) {
        return 275;
    } else if (getMaxStepCnt() == 8) {
        return 538;
    } else {
        //527-279
        return 1058;
    }
}

int getIcNum() {
    return icNumber;
}

/**
 * 灯光同色场景
 */
vector<int> getSameColorVector() {
    vector<int> sameColorScore;
    if (getMaxStepCnt() == 7) {
        sameColorScore = {148, 275};
    } else if (getMaxStepCnt() == 8) {
        sameColorScore = {283, 538};
    } else {
        sameColorScore = {547, 1058};
    }
    return sameColorScore;
}

/**
 * 根据得分计算灯序号
 */
void setLightIndex(LightPoint &point, int score) {
    int scoreP = score;
    int lightIndex = 0;
    vector<int> sameColorVector = getSameColorVector();
    int scoreMin = getScoreMin();
    if (scoreP > sameColorVector[1]) {
        lightIndex = scoreP - scoreMin - 2;
    } else {
        lightIndex = scoreP - scoreMin - 1;
    }
    lightIndex = lightIndex * 2;
    point.lightIndex = lightIndex;
}

int getLightIndex(int score) {
    int scoreP = score;
    int lightIndex = 0;
    vector<int> sameColorVector = getSameColorVector();
    int scoreMin = getScoreMin();
    if (scoreP > sameColorVector[1]) {
        lightIndex = scoreP - scoreMin - 2;
    } else {
        lightIndex = scoreP - scoreMin - 1;
    }
    lightIndex = lightIndex * 2;
    return lightIndex;
}



