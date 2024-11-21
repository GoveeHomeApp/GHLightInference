#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

//#include "common.hpp"
#include "discoverer.hpp"
#include <map>
using namespace cv;
using namespace std;

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
     * 坐标集合
     */
    vector<Point2f> pointXys;

    /**
     * 正常序列点位
     */
    vector<LightPoint> normalPoints;
    map<int, vector<LightPoint>> sameSerialNumMap;
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
LightPoint
meanLightColor(const Mat &src, const vector<vector<cv::Point>> &contours, int frameStep,
               const LightPoint &lPoint, Mat &meanColorMat,
               double forceRadius = 8.0);

/**
 * 对灯带光点排序
 */
LampBeadsProcessor sortLampBeads(Mat &src, vector<Mat> &outMats, vector<Point2f> &trapezoid4Points);

/**
 * 获取区域颜色集合
 */
vector<LightPoint>
findColorType(const Mat &src, int stepFrame, const vector<LightPoint> &points,
              vector<Mat> &outMats);

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
string lightPointsToJson(const vector<LightPoint> &points, int lightTypeSet);


/**
 * 计算点位平均距离
 */
double calculateAverageDistance(LampBeadsProcessor &processor);

/**
 * 推测中间夹点
 */
vector<LightPoint>
decisionCenterPoints(vector<LightPoint> &input, double averageDistance);

vector<LightPoint>
decisionCenterPoints2(const vector<LightPoint> &input, double averageDistance);

/**
 * 从红绿固定点和错点中推测点位
 */
void decisionRightLeftPoints(vector<LightPoint> &totalPoints, bool findFromError);

/**处理剩余无序点位*/
void decisionRemainingPoints(LampBeadsProcessor &processor);

void release();

/**
 * Point2i集合输出json
 */
string point2iToJson(const vector<Point2f> &points);

string splicedJson(string a, string b);

bool isApproximatelyHorizontal(Point2f A, Point2f B, Point2f C);

LightPoint inferredAB2Next(LightPoint &A, LightPoint &B, bool findErrorPoints);

bool compareIndex(const LightPoint &p1, const LightPoint &p2);

/**
 * 找出最可能点位
 */
LightPoint findLamp(Point2f &center, double minDistance, bool checkDistance, int inferredLightIndex,
                    bool findErrorPoints, bool erase = true);

/**
 * 从集合中查找点位
 */
LightPoint findLampInVector(Point2f &center, double minDistance, bool checkDistance,
                            vector<LightPoint> &points, int type, bool erase = false);

/**
 * 根据水平方向推断右边点
 */
LightPoint inferredRight(LightPoint &curLPoint,
                         LightPoint &lastLPoint,
                         LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                         bool findErrorPoints);

/**
 * 根据水平方向推断左边点
 */
LightPoint inferredLeft(LightPoint &curLPoint,
                        LightPoint &lastLPoint,
                        LightPoint &nextLPoint, int i, vector<LightPoint> &totalPoints,
                        bool findErrorPoints);

vector<LightPoint> fillMissingPoints(const vector<LightPoint> &totalPoints, double avgDistance);

/**
 * 推测中间点
 * @param A 后一个点
 * @param B 前一个点
 */
LightPoint
inferredCenter(double avgDistance, LightPoint &A, LightPoint &B, bool findErrorPoints);


vector<LightPoint>
mergeOverlappingPoints(const vector<LightPoint> &points, float radius = 4.0,
                       float overlapThreshold = 0.7);

bool reCheckScore(vector<LightPoint> lightPoints);

Rect2i safeRect2i(const Rect2i &region, const cv::Size &imageSize);

Mat
buildRect(const LightPoint lp, const Mat &src, int forceRadius = 0);

Mat
buildRect(const Point2f position, const Mat &src, int forceRadius);

#endif
