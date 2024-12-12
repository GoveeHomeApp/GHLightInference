#ifndef ColorCodingH61DX_HPP
#define ColorCodingH61DX_HPP

#include <vector>

/// H61DX颜色编码
class ColorCodingH61DX
{
public:
    /// @brief ic个数
    int icCount;

    /// @brief 构造函数
    /// @param icCount ic个数
    ColorCodingH61DX(int icCount = 70);

    /// @brief 灯效展示颜色序列
    std::vector<int> getDetectionColors();

    /// @brief 获取起始的颜色序列
    std::vector<int> getStartColors();

private:
    /// @brief 编码宽度
    int _bitWidth;
    /// @brief 编码个数
    int _count;

    /// @brief 获取对应序号的颜色编码
    /// @param index 编码序号
    /// @return 对应的颜色编码
    std::vector<int> getIndexColors(int index);
};

#endif // ColorCodingH61DX_HPP
