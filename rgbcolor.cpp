#include "rgbcolor.h"

RGBColor::RGBColor()
    :r(.0f), g(.0f), b(.0f)
{

}

RGBColor::RGBColor(float red, float green, float blue)
    : r(red), g(green), b(blue) {

}

RGBColor::RGBColor(real brightness)
    : r(brightness), g(brightness), b(brightness){

}

RGBColor::RGBColor(const RGBColor &color)
    : r(color.r), g(color.g), b(color.b) {

}

RGBColor::RGBColor(const QRgb &color)
    : r((real)((color & 0x00FF0000) >> 16)/255.0),
      g((real)((color & 0x0000FF00) >> 8)/255.0),
      b((real)((color & 0x000000FF)/255.0))
{

}


RGBColor RGBColor::truncate() const
{
//    float r_ = r < 1.0f ? r : 1.0f;
//    float g_ = g < 1.0f ? g : 1.0f;
//    float b_ = b < 1.0f ? b : 1.0f;

//    return RGBColor(r_, g_, b_);

//    return RGBColor(clamp(r, .0f, 1.0f),
//                    clamp(g, .0f, 1.0f),
//                    clamp(b, .0f, 1.0f));
    float max = std::max(r, std::max(g,b));
    if (max > 1.0)
        return (*this) / max;
    return *this;
}

RGBColor RGBColor::operator+(const RGBColor &color)
{
    return RGBColor(this->r + color.r, this->g + color.g, this->b + color.b);
}

RGBColor RGBColor::operator*(real f)
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

RGBColor &RGBColor::operator/=(real f)
{
    r /= f;
    g /= f;
    b /= f;

    return *this;
}

RGBColor &RGBColor::operator*=(real f)
{
    r *= f;
    g *= f;
    b *= f;

    return *this;
}

RGBColor &RGBColor::operator*=(const RGBColor &c)
{
    r *= c.r;
    g *= c.g;
    b *= c.b;

    return *this;
}



RGBColor operator/(const RGBColor &c, float f)
{
    return RGBColor(c.r / f, c.g / f, c.b / f);
}

RGBColor operator*(const RGBColor &a, const RGBColor &b)
{
    return RGBColor(a.r * b.r, a.g * b.g, a.b * b.b);
}

real clamp(real x, real min, real max)
{
    return x < min ? min : (x > max ? max : x);
}
