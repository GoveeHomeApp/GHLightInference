#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <map>

#define LOG_TAG "log_detection_"

#define LOGV(TAG, ...) printf(__VA_ARGS__)
#define LOGD(TAG, ...) printf(__VA_ARGS__)
#define LOGW(TAG, ...) printf(__VA_ARGS__)
#define LOGE(TAG, ...) printf(__VA_ARGS__)

#include <time.h>
#endif

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

class LightPoint {
public:
    cv::Point2i point2f;
    double with = 7.0;
    double height = 7.0;
    int score = -1;
    double brightness = -1;
    int lightIndex = -1;
    CUS_COLOR_TYPE type = E_W;
    LIGHT_STATUS errorStatus = NORMAL;
    float tfScore = 0;
    cv::Rect tfRect;
    cv::RotatedRect rotatedRect;
public:
    ~LightPoint() {
        // 析构函数，释放资源
    }

    LightPoint() {
    }


    LightPoint(cv::Point2f point, double withSet, double heightset) {
        point2f = point;
        with = withSet;
        height = heightset;
    }

    LightPoint(LIGHT_STATUS errorStatusSet) {
        this->errorStatus = errorStatusSet;
    }

    LightPoint copyPoint(CUS_COLOR_TYPE colorType, cv::Scalar scalar) {
        LightPoint point = LightPoint(this->point2f, this->with, this->height);
        point.type = colorType;
        point.brightness = scalar[0];
        point.rotatedRect = this->rotatedRect;
        return point;
    }

    cv::Mat buildRect(cv::Mat &src, cv::Rect &roi) {
        if (src.empty()) {
            LOGE(LOG_TAG, "buildRect src is empty!");
        }
        int x = point2f.x; // 指定坐标x
        int y = point2f.y; // 指定坐标y
        roi = cv::Rect();
        if (x - with / 2 < 0) {
            roi.x = 1;
        } else {
            roi.x = x - with / 2 + 1;
        }
        if (y - height / 2 < 0) {
            roi.y = 1;
        } else {
            roi.y = y - height / 2 + 1;
        }
        if (roi.x + roi.width > src.cols) {
            LOGE(LOG_TAG, "x>cols with:%d  src-cols: %d   x: %d ", (roi.x + roi.width), src.cols,
                 roi.x);
            roi.width = src.cols - roi.x - 1;
        }

        if (roi.y + roi.height > src.rows) {
            LOGE(LOG_TAG, "y>rows height:%d  src-rows: %d", (roi.y + roi.height), src.rows);
            roi.height = src.rows - roi.y - 1;
        }
        roi.width = this->with;
        roi.height = this->height;
        if (roi.width < 5) roi.width = 5;
        if (roi.height < 5) roi.height = 5;
//        LOGD(LOG_TAG, "roi = %d x %d, w = %d, h = %d, src = %d x %d", roi.x, roi.y, roi.width,
//             roi.height, src.cols, src.rows);
        cv::Mat region = src(roi);
        return region;
    }

    // 重载==运算符以支持比较
    bool operator==(const LightPoint &other) const {
        return lightIndex == other.lightIndex && point2f.x == other.point2f.x &&
               point2f.y == other.point2f.y;
    }
};
