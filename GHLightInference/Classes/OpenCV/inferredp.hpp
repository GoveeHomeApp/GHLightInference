#ifndef __INFERREDP_HPP__
#define __INFERREDP_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;

/**
 * 查找最接近的中心点
 * @param lA
 * @param lB
 * @param points
 * @param targetDistance
 * @return
 */
LightPoint findMostLikelyCenter(const LightPoint &lA, const LightPoint &lB, int inferredLightIndex,
                                int lightType,
                                unordered_map<int, vector<LightPoint>> &sequenceTypeMap,
                                double targetDistance);

LightPoint
findExtensionPointAB2C(const LightPoint &lA, const LightPoint &lB, int inferredLightIndex,
                       int lightType,
                       unordered_map<int, vector<LightPoint>> sequenceTypeMap,
                       double targetDistance);

double sigmoid(double x, double scale = 10.0);

double smoothLimit(double value, double min, double max, double transitionRange = 0.1);
#endif
