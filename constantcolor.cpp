#include "constantcolor.h"

ConstantColor::ConstantColor()
    : color(0, 0, 0)
{

}

ConstantColor::ConstantColor(const RGBColor &color_)
    : color(color_)
{

}

RGBColor ConstantColor::get_color(const ShadeRec &sr)
{
    return color;
}
