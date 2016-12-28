#include "rgbcolor.h"

RGBColor::RGBColor()
    :r(.0f), g(.0f), b(.0f)
{

}

RGBColor::RGBColor(float red, float green, floart blue)
    : r(red), g(green), b(blue) {

}

RGBColor RGBColor::operator+(const RGBColor &color)
{
    return RGBColor(this->r + color.r, this->g + color.g, this->b + color.b);
}

RGBColor RGBColor::operator*(float f)
{
    return RGBColor(this->r * f, this->g * f, this->b * f);
}
