#include "ambient.h"

#include "rgbcolor.h"
#include "vector.h"

Ambient::Ambient()
    :ls(1.0), color(1.0)
{
    shadows = false;
}


Ambient::Ambient(float brightness, float r, float g, float b)
    : ls(brightness), color(r, g, b) {
    shadows = false;
}

Vector Ambient::get_direction(ShadeRec &sr)
{
    return Vector();
}

RGBColor Ambient::L(ShadeRec &sr)
{
    return ls * color;
}

bool Ambient::in_shadow(Ray& ray, ShadeRec& sr)
{
    return false;
}
