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
<<<<<<< HEAD
const RGBColor operator* (const RGBColor& a, const RGBColor& b);
=======
>>>>>>> 5eababce84a924f7b3f281471cc8115b09966a0d


Q_DECLARE_METATYPE(RGBColor)

#endif // RGBCOLOR_H
