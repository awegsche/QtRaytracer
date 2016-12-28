#ifndef RGBCOLOR_H
#define RGBCOLOR_H


class RGBColor
{
public:
    float r;
    float g;
    float b;

public:
    RGBColor();
    RGBColor(float red, float green, floart blue);

    RGBColor operator+ (const RGBColor& color);
    RGBColor operator* (float f);
};

#endif // RGBCOLOR_H
