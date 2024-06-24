#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

#include "common.hpp"
#include "discoverer.hpp"

using namespace cv;
using namespace std;

/**补全等级*/
enum COMPLETION_LEVEL {
    /**仅做tf识别到的点补全*/
    COMPLETION_INTERVAL = 0,

    /**对tf识别到的点以及单个间隔点补全*/
    COMPLETION_INTERVAL_INFERRED = 2,

    /**根据线性方向进行全量补全*/
    COMPLETION_LINE_ALL = 2
};

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
    SAME_COLOR = 1,
    SAME_INDEX = 2,
    OUT_BOUNDARY_MIN = 3,
    OUT_BOUNDARY_MAX = 4,
    INFERRED_EMPTY_POINT = 5,
    INFERRED_POINT = 6,
    EMPTY_POINT = 7

};

class LightPoint {
public:
    Point2i point2f;
    double with = 7.0;
    double height = 7.0;
    int score = -1;
    double brightness = -1;
    int lightIndex = -1;
    CUS_COLOR_TYPE type = E_W;
    LIGHT_STATUS errorStatus = NORMAL;
    float tfScore = 0;
    Rect_<int> tfRect;
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

    LightPoint copyPoint(CUS_COLOR_TYPE colorType, Scalar scalar) {
        LightPoint point = LightPoint(this->point2f, this->with, this->height);
        point.type = colorType;
        point.brightness = scalar[0];
        return point;
    }

    Mat buildRect(Mat &src, cv::Rect &roi) {
        int x = point2f.x; // 指定坐标x
        int y = point2f.y; // 指定坐标y
        //Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
        roi = cv::Rect(x - with / 2, y - height / 2, with, height);

        if (x - with / 2 < 0) {
            roi.x = 0;
            LOGE(LOG_TAG, "x<0");
        }
        if (y - height / 2 < 0) {
            roi.y = 0;
            LOGE(LOG_TAG, "y<0");
        }
        if (roi.x + roi.width > src.cols) {
            LOGE(LOG_TAG, "x>cols with:%d  src-cols: %d   x: %d ", (roi.x + roi.width), src.cols,
                 roi.x);
            roi.width = src.cols - roi.x;
        }
        if (roi.y + roi.height > src.rows) {
            LOGE(LOG_TAG, "y>rows height:%d  src-rows: %d", (roi.y + roi.height), src.rows);
            roi.height = src.rows - roi.y;
        }
        Mat region = src(roi);
        return region;
    }
};

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @param outMats 输出流程中的测试图像
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat, vector<Mat> &outMats);

/**
 * 根据定义好的步骤进行灯带排序
 * @param frameStep 当前轮数
 * @param resultObjects 当前tf识别的结果以及opencv找色的结果
 * @param outMats 输出流程中的测试图像
 * @return
 */
String
sortStripByStep(int frameStep, vector<LightPoint> &resultObjects, int lightType,
                vector<Mat> &outMats);


/**
 * 获取区域 hsv 色相
 */
LightPoint meanColor(Mat &src, int stepFrame, LightPoint &lPoint, Mat &meanColorMat);

/**
 * 对灯带光点排序
 */
vector<LightPoint>
checkLightStrip(Mat &src, vector<Mat> &outMats, vector<Point2i> &trapezoid4Points);

Mat alignImg(Mat &src, Mat &trans, bool back4Matrix);

/**
 * 获取区域颜色集合
 */
vector<LightPoint>
findColorType(Mat &src, int stepFrame, vector<LightPoint> &points, vector<Mat> &outMats);

/**
 * 从小到大排序
 */
bool compareScore(const LightPoint &p1, const LightPoint &p2);

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LightPoint> &points);

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2i> &points);

string splicedJson(string a, string b);

bool isApproximatelyHorizontal(Point2i A, Point2i B, Point2i C);

LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, vector<LightPoint> &redSameVector,
                           vector<LightPoint> &greenSameVector);

bool compareIndex(const LightPoint &p1, const LightPoint &p2);

LightPoint syncRectPoints(Point2i &center, double minDistance,
                          vector<LightPoint> &points);

LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, vector<LightPoint> &pointsNew,
                         vector<LightPoint> &redSameVector, vector<LightPoint> &greenSameVector);

LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, vector<LightPoint> &pointsNew,
                        vector<LightPoint> &redSameVector, vector<LightPoint> &greenSameVector);

#endif
