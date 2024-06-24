#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

#include "common.hpp"
#include "discoverer.hpp"

using namespace cv;
using namespace std;

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
        if (src.empty()) {
            LOGE(LOG_TAG, "buildRect src is empty!");
        }
        int x = point2f.x; // 指定坐标x
        int y = point2f.y; // 指定坐标y
        //Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
        roi = cv::Rect();
        //  x - with / 2, y - height / 2, with, height
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
        if (roi.width < 5) roi.width = 5;
        if (roi.height < 5) roi.height = 5;
//        LOGD(LOG_TAG, "roi = %d x %d, w = %d, h = %d, src = %d x %d", roi.x, roi.y, roi.width,
//             roi.height, src.cols, src.rows);
        Mat region = src(roi);
        return region;
    }

    // 重载==运算符以支持比较
    bool operator==(const LightPoint &other) const {
        return lightIndex == other.lightIndex && point2f.x == other.point2f.x &&
               point2f.y == other.point2f.y;
    }
};

/**
 * 记录当前处理流程中的状态集合
 */
class LampBeadsProcessor {
public:
    int scoreMin;
    int scoreMax;
    int maxFrameStep;
    int averageDistance;

    /**
     * 同色得分点
     * key:得分
     * value：index集合
     */
    map<int, vector<int>> sameSerialNumMap;
    /**
     * 红色非序列灯珠集合
     */
    vector<LightPoint> redSameVector;

    /**
     * 坐标集合
     */
    vector<Point2i> pointXys;

    /**
     * 绿色非序列灯珠集合
     */
    vector<LightPoint> greenSameVector;
    /**
    * 得分错点集合
    */
    vector<LightPoint> errorSerialVector;
    /**
     * 正常序列点位
     */
    vector<LightPoint> normalPoints;
    /**
      * 所有点位集合
      */
    vector<LightPoint> totalPoints;
public:
    ~LampBeadsProcessor() {
    }


    LampBeadsProcessor() {
    }

    LampBeadsProcessor(int scoreMin, int scoreMax, int maxFrameStep) {
        this->scoreMin = scoreMin;
        this->scoreMax = scoreMax;
        this->maxFrameStep = maxFrameStep;
    }
};

/**
 * 统计所有得分
 * @param processor
 */
int statisticalScorePoints(Mat &src, vector<Mat> &outMats, LampBeadsProcessor &processor);

/**
 * 对齐并输出640正方形图像
 * @param frameStep 当前轮数
 * @param originalMat 输入原图
 * @return
 */
Mat alignResize(int frameStep, Mat &originalMat);

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
sortLampBeads(Mat &src, vector<Mat> &outMats, vector<Point2i> &trapezoid4Points);

Mat alignImg(Mat &src, Mat &trans, bool back4Matrix);

/**
 * 获取区域颜色集合
 */
vector<LightPoint>
findColorType(Mat &src, int stepFrame, vector<LightPoint> &points, vector<Mat> &outMats);

/**删除不连续错点*/
void deleteDiscontinuousPoints(LampBeadsProcessor &processor);

/**
 * 删除离群点+构建梯形
 */
void deleteEstablishGroupPoints(Mat &src, vector<Mat> &outMats);

/**
 * 从小到大排序
 */
bool compareScore(const LightPoint &p1, const LightPoint &p2);

/**
 * LightPoint集合输出json
 */
string lightPointsToJson(const vector<LightPoint> &points);

/**
 * 处理同色得分点
 * @param samePoints
 */
void
decideSameScorePoint(LampBeadsProcessor &processor, Mat &src, vector<Mat> &outMats);

/**
 * 计算点位平均距离
 */
double calculateAverageDistance(LampBeadsProcessor &processor);

/**
 * 推测中间夹点
 */
void
decisionCenterPoints(LampBeadsProcessor &processor, Mat &src);

/**
 * 从红绿固定点和错点中推测点位
 */
void decisionRightLeftPoints(LampBeadsProcessor &processor);

/**处理剩余无序点位*/
void decisionRemainingPoints(LampBeadsProcessor &processor);

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2i> &points);

string splicedJson(string a, string b);

bool isApproximatelyHorizontal(Point2i A, Point2i B, Point2i C);

LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, LampBeadsProcessor &processor);

bool compareIndex(const LightPoint &p1, const LightPoint &p2);

/**
 * 找出最可能点位
 */
LightPoint findLamp(Point2i &center, double minDistance, bool checkDistance, int inferredLightIndex,
                    LampBeadsProcessor &processor);

/**
 * 从集合中查找点位
 */
LightPoint findLampInVector(Point2i &center, double minDistance, bool checkDistance,
                            vector<LightPoint> &points);

/**
 * 根据水平方向推断右边点
 */
LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, LampBeadsProcessor &processor);

/**
 * 根据水平方向推断左边点
 */
LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, LampBeadsProcessor &processor);

/**
 * 推测中间点
 * @param A 后一个点
 * @param B 前一个点
 */
LightPoint inferredCenter(LampBeadsProcessor &processor, LightPoint &A, LightPoint &B);

#endif
