#ifndef __SELECT_HPP__
#define __SELECT_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>
#include <map>

using namespace cv;
using namespace std;

void
processSamePoints(Mat &src, vector<Mat> &outMats, vector<LightPoint> &totalPoints,
                  vector<LightPoint> &errorPoints,
                  float avgDistance, const map<int, vector<LightPoint>> sameSerialNumMap,
                  int lightType);

#endif
