#include "constantcolor.h"

ConstantColor::ConstantColor()
    : color(0, 0, 0)
{

}

ConstantColor::ConstantColor(const RGBColor &color_)
    : color(color_)
{

}

ConstantColor::ConstantColor(const real r, const real g, const real b)
    :color(RGBColor(r, g, b))
{

}

RGBColor ConstantColor::get_color(const ShadeRec &sr)
{
    return color;
}
