#ifndef MATERIAL_H
#define MATERIAL_H

#include "constants.h"

class RGBColor;
class ShadeRec;

class Material
{
public:
    bool has_transparency;
public:
    Material();

    virtual RGBColor shade(ShadeRec& sr);
    virtual RGBColor area_light_shade(ShadeRec& sr);
    virtual RGBColor path_shade(ShadeRec& sr);
    virtual RGBColor noshade(ShadeRec& sr);
    virtual real transparency(const ShadeRec& sr);
};

#endif // MATERIAL_H
