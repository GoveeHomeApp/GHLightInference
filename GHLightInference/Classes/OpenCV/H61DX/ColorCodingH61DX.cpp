#include "ColorCodingH61DX.hpp"

using namespace std;

namespace
{
    const int COLOR_RED = 0xFF0000;
    const int COLOR_GREEN = 0x00FF00;
    const int COLOR_YELLOW = 0xFFFF00;
    const int COLOR_BLUE = 0x0000FF;
    const int COLOR_UNIT_COUNT = 3;
    const int START_IC_COUNT = COLOR_UNIT_COUNT;

    int getBitWidth(int icNum)
    {
        if (icNum <= START_IC_COUNT)
        {
            return 1;
        }
        auto length = icNum - START_IC_COUNT;
        auto bitWidth = 1;
        while (pow(2, bitWidth) * bitWidth * COLOR_UNIT_COUNT < length)
        {
            bitWidth++;
        }
        return bitWidth;
    }

    int getCount(int icNum, int bitWidth)
    {
        if (icNum <= START_IC_COUNT)
        {
            return 0;
        }
        auto length = icNum - 3;
        return length / bitWidth + (length % bitWidth > 0 ? 1 : 0);
    }
}

ColorCodingH61DX::ColorCodingH61DX(int icCount) : icCount(icCount)
{
    _bitWidth = getBitWidth(icCount);
    _count = getCount(icCount, _bitWidth);
}

// 获取灯效颜色
std::vector<int> ColorCodingH61DX::getDetectionColors()
{
    auto result = vector<int>();
    auto start = this->getStartColors();
    result.insert(result.end(), start.begin(), start.end());
    for (int i = 0; i < _count; i++)
    {
        auto colors = this->getIndexColors(i);
        result.insert(result.end(), colors.begin(), colors.end());
    }
    // 将result长度限定在icCount内
    result.resize(icCount);
    return result;
}

// 获取开始颜色
std::vector<int> ColorCodingH61DX::getStartColors()
{
    return {COLOR_YELLOW, COLOR_GREEN, COLOR_YELLOW};
}

std::vector<int> ColorCodingH61DX::getIndexColors(int index)
{
    auto colors = vector<int>();
    // 按位遍历index的每一位，如果为0则添加绿色，为1则添加黄色
    for (int i = 0; i < _bitWidth; i++)
    {
        colors.push_back(COLOR_RED);
        if ((index >> i) & 0x01)
        {
            colors.push_back(COLOR_YELLOW);
        }
        else
        {
            colors.push_back(COLOR_GREEN);
        }
        colors.push_back(COLOR_BLUE);
    }
    return colors;
}
