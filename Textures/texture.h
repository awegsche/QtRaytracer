#ifndef TEXTURE_H
#define TEXTURE_H

#include "rgbcolor.h"
#include "shaderec.h"

class Texture
{
public:
    Texture();

    virtual RGBColor get_color(const ShadeRec& sr) = 0;
    virtual real get_transparency(const ShadeRec& sr);
};

#endif // TEXTURE_H
