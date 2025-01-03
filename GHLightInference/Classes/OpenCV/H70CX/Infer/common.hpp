#pragma once
#ifdef NO
#undef NO
#define NO MY_NO
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "LogUtils.h"
#include "color_splitter.hpp"
#include <time.h> // clock_gettime

// 全局ColorSplitter声明
extern ColorSplitter g_colorSplitter;

enum CUS_COLOR_TYPE {
    E_W = 0,
    E_RED = 1,
    E_GREEN = 2,
    E_BLUE = 3,
};

const int STEP_VALID_FRAME_START = 0;
const int TYPE_H70CX_3D = 0;

class TfPoint {
public:
    ~TfPoint() {
        // 析构函数，释放资源
    }

    TfPoint() {
    }

    cv::Point2f position;
    cv::Rect2i rect;
    bool isOutside = false;

    std::string toJson() const {
        return "{"
               "\"position\": {"
               "\"x\": " + std::to_string(position.x) + ", "
                                                        "\"y\": " + std::to_string(position.y) +
               "}, "
               "\"isOutside\": " + (isOutside ? "true" : "false") +
               "}";
    }


};

class LightPoint {
public:
    cv::Point2f position;
    int label = -1;
    double with = 7.0;
    double height = 7.0;
    int score = -1;
    double brightness = -1;
    CUS_COLOR_TYPE type = E_W;
    float tfScore = 0;
    cv::Rect2i tfRect;
    cv::RotatedRect rotatedRect;
    std::vector<int> neighbors;
    float localDensity;  // 新增：局部密度
public:
    ~LightPoint() {
        // 析构函数，释放资源
    }

    LightPoint() {
    }


    LightPoint(cv::Point2f point, double withSet, double heightset) {
        position = point;
        with = withSet;
        height = heightset;
    }

    LightPoint(cv::Point2f point, double withSet, double heightset, CUS_COLOR_TYPE type) {
        position = point;
        with = withSet;
        height = heightset;
        type = type;
    }

    LightPoint(int labelSet) {
        label = labelSet;
    }

    // 重载==运算符以支持比较
    bool operator==(const LightPoint &other) const {
        return label == other.label && position.x == other.position.x &&
               position.y == other.position.y;
    }
};

