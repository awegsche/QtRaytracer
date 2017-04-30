#ifndef CONSTANTCOLOR_H
#define CONSTANTCOLOR_H

#include "texture.h"
#include "rgbcolor.h"

class ConstantColor : public Texture
{
private:
    RGBColor color;
public:
    ConstantColor();
    ConstantColor(const RGBColor& color_);

    // Texture interface
public:
    RGBColor get_color(const ShadeRec &sr);
};

#endif // CONSTANTCOLOR_H
