#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

//#include "common.hpp"
#include "discoverer.hpp"

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
     * 同色得分点
     * key:得分
     * value：index集合
     */
    map<int, vector<int>> sameSerialNumMap;


    /**
     * 坐标集合
     */
    vector<Point2i> pointXys;

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
LampBeadsProcessor sortLampBeads(Mat &src, vector<Mat> &outMats, vector<Point2i> &trapezoid4Points);

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

void release();

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
                            vector<LightPoint> &points, int type);

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
