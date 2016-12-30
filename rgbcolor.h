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
    RGBColor(float brightness);

    RGBColor truncate() const;


    RGBColor operator+ (const RGBColor& color);
    RGBColor operator* (float f);
    RGBColor& operator+= (const RGBColor& c);
    RGBColor& operator/= (float f);
    RGBColor& operator*= (float f);
};

const RGBColor operator/ (const RGBColor& c, float f);
const RGBColor operator* (const RGBColor& a, const RGBColor& b);



Q_DECLARE_METATYPE(RGBColor)

#endif // RGBCOLOR_H
