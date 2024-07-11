#ifndef __INTERPOLATE70CX_HPP__
#define __INTERPOLATE70CX_HPP__

#include "common.hpp"
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;

struct Group {
    int start;
    int end;
    int count;
    bool isKnown;
    vector<LightPoint> points;

    Group(int s, int e, bool k) : start(s), end(e), count(e - s + 1), isKnown(k) {}
};

struct GapInfo {
    int start;
    int end;
    double ratio;
};

vector<LightPoint> completeLightPoints2D(const vector<LightPoint> &inputPoints, int maxNum);

vector<LightPoint> interpolatePoints3D(const vector<LightPoint> &points);

vector<GapInfo> analyzeGaps(vector<Group> &groups, double threshold = 1.7);

vector<Group> groupLightPoints(vector<LightPoint> &lightPoints);

#endif
