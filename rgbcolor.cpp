#include "rgbcolor.h"

RGBColor::RGBColor()
    :r(.0f), g(.0f), b(.0f)
{

}

RGBColor::RGBColor(float red, float green, float blue)
    : r(red), g(green), b(blue) {

}

RGBColor::RGBColor(float brightness)
    : r(brightness), g(brightness), b(brightness){

}


RGBColor RGBColor::truncate() const
{
    float r_ = r < 1.0f ? r : 1.0f;
    float g_ = g < 1.0f ? g : 1.0f;
    float b_ = b < 1.0f ? b : 1.0f;

    return RGBColor(r_, g_, b_);
}

RGBColor RGBColor::operator+(const RGBColor &color)
{
    return RGBColor(this->r + color.r, this->g + color.g, this->b + color.b);
}

RGBColor RGBColor::operator*(float f)
{
    return RGBColor(this->r * f, this->g * f, this->b * f);
}

RGBColor &RGBColor::operator+=(const RGBColor &c)
{
    r += c.r;
    g += c.g;
    b += c.b;

    return *this;
}

RGBColor &RGBColor::operator/=(float f)
{
    r /= f;
    g /= f;
    b /= f;

    return *this;
}

RGBColor &RGBColor::operator*=(float f)
{
    r *= f;
    g *= f;
    b *= f;

    return *this;
}



const RGBColor operator/(const RGBColor &c, float f)
{
    return RGBColor(c.r / f, c.g / f, c.b / f);
}

const RGBColor operator*(const RGBColor &a, const RGBColor &b)
{
    return RGBColor(a.r * b.r, a.g * b.g, a.b * b.b);
}
