#pragma once

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
    TYPE_H682X = 2,
};

enum FIND_TYPE {
    TYPE_SEQUENCE = 0,
    TYPE_NO_SEQUENCE = 1,
    TYPE_ERROR = 2,
};

class LightPoint {
public:
    cv::Point2f position;
    double with = 9.0;
    double height = 9.0;
    int score = -1;
    double brightness = -1;
    int label = -1;
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

    cv::Rect2i safeRect(const cv::Rect2i &region, const cv::Size &imageSize) {
        cv::Rect2i safe = region;
        safe.x = safe.x;
        safe.y = safe.y;
        safe.width = safe.width;
        safe.height = safe.height;
        safe.x = std::max(0, std::min(safe.x, imageSize.width - 1));
        safe.y = std::max(0, std::min(safe.y, imageSize.height - 1));
        safe.width = std::min(safe.width, imageSize.width - safe.x);
        safe.height = std::min(safe.height, imageSize.height - safe.y);
        return safe;
    }

    cv::Mat buildRect(cv::Mat &src, cv::Rect &roi) {
        cv::Mat region;
        try {
            if (src.empty()) {
                LOGE(LOG_TAG, "buildRect src is empty!");
            }
            float x = position.x; // 指定坐标x
            float y = position.y; // 指定坐标y
            roi = cv::Rect(x, y, with, height);
            if (roi.width < 9.0) roi.width = 9.0;
            if (roi.height < 9.0) roi.height = 9.0;
            cv::Rect2i safeR = safeRect(roi, src.size());
            region = src(safeR);
        } catch (...) {
            roi = cv::Rect();
            LOGE(LOG_TAG, "构建点的矩形失败");
        }
        return region;
    }

    // 重载==运算符以支持比较
    bool operator==(const LightPoint &other) const {
        return label == other.label && position.x == other.position.x &&
               position.y == other.position.y;
    }
};
