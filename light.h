#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"

class Vector;
class RGBColor;
class ShadeRec;

class Light
{

public:
    Light();

    virtual Vector get_direction(ShadeRec& sr) = 0;
    virtual RGBColor L(ShadeRec& sr) = 0;

    bool casts_shadows();
    void set_shadows(bool b);
    virtual bool in_shadow(Ray& ray, ShadeRec& sr) = 0;

protected:
    bool shadows;
};

#endif // LIGHT_H
