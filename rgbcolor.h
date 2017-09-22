#ifndef RGBCOLOR_H
#define RGBCOLOR_H

#include <QMetaType>
#include "constants.h"
#include <QRgb>

class RGBColor
{

public:
    real r;
    real g;
    real b;

public:
    RGBColor();
    RGBColor(real red, real green, real blue);
    RGBColor(real brightness);
    RGBColor(const RGBColor& color);
    RGBColor(const QRgb& color);

    RGBColor truncate() const;


    RGBColor operator+ (const RGBColor& color);
    RGBColor operator* (real f);
    RGBColor& operator+= (const RGBColor& c);
    RGBColor& operator/= (real f);
    RGBColor& operator*= (real f);
    RGBColor& operator*= (const RGBColor& c);

    uint to_uint() const;
};

RGBColor operator/ (const RGBColor& c, real f);
RGBColor operator* (const RGBColor& a, const RGBColor& b);

real clamp(real x, real min, real max);

Q_DECLARE_METATYPE(RGBColor)

#endif // RGBCOLOR_H
