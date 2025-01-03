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
#include "logger.hpp"

#define LOG_TAG "OpenCv"
#define TAG_INFERRED "OpenCv_Inferred"
#define TAG_DELETE "OpenCv_Delete"
#define TAG_ADD "OpenCv_Add"

#include <time.h> // clock_gettime


enum CUS_COLOR_TYPE {
    E_W = 0,
    E_RED = 1,
    E_GREEN = 2,
    E_BLUE = 3,
};
enum STEP_FRAME {
    STEP_VALID_FRAME_START = 0,
};
enum LIGHT_STATUS {
    NORMAL = 0,
    EMPTY_POINT = 1,
    ERASE_POINT = 2
};
enum LIGHT_TYPE {
    TYPE_H70CX_3D = 0,
    TYPE_H70CX_2D = 1,
};

enum FIND_TYPE {
    TYPE_SEQUENCE = 0,
    TYPE_NO_SEQUENCE = 1,
    TYPE_ERROR = 2,
};

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
    LIGHT_STATUS errorStatus = NORMAL;
    float tfScore = 0;
    cv::Rect2i tfRect;
    cv::RotatedRect rotatedRect;
    std::vector<int> neighbors;
    float localDensity;  // 新增：局部密度
    cv::Point startPoint;  //
    cv::Point endPoint;  //
    bool isInterpolate = false;
    FIND_TYPE findType = TYPE_SEQUENCE;
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

    LightPoint(LIGHT_STATUS errorStatusSet) {
        this->errorStatus = errorStatusSet;
    }

    LightPoint copyPoint(CUS_COLOR_TYPE colorType, cv::Scalar scalar) {
        LightPoint point = LightPoint(this->position, this->with, this->height);
        point.type = colorType;
        point.brightness = scalar[0];
        point.rotatedRect = this->rotatedRect;
        return point;
    }


    // 重载==运算符以支持比较
    bool operator==(const LightPoint &other) const {
        return label == other.label && position.x == other.position.x &&
               position.y == other.position.y;
    }
};

