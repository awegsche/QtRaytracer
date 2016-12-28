#ifndef RGBCOLOR_H
#define RGBCOLOR_H

#include <QMetaType>

class RGBColor
{

public:
    float r;
    float g;
    float b;

public:
    RGBColor();
    RGBColor(float red, float green, float blue);

    RGBColor operator+ (const RGBColor& color);
    RGBColor operator* (float f);
};

Q_DECLARE_METATYPE(RGBColor)

#endif // RGBCOLOR_H
