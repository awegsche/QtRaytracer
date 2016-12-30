#ifndef MATERIAL_H
#define MATERIAL_H

class RGBColor;
class ShadeRec;

class Material
{
public:
    Material();

    virtual RGBColor shade(ShadeRec& sr);
    virtual RGBColor area_light_shade(ShadeRec& sr);
    virtual RGBColor path_shade(ShadeRec& sr);
};

#endif // MATERIAL_H
