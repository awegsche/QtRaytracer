#ifndef GEOMETRICOBJECT_H
#define GEOMETRICOBJECT_H

#include "constants.h"
#include "rgbcolor.h"

class Ray;
class ShadeRec;


class GeometricObject
{
public:
    RGBColor color;
public:
    GeometricObject();

    virtual bool hit(const Ray& ray, real& tmin, ShadeRec &sr) const = 0;

    RGBColor get_color();
};

#endif // GEOMETRICOBJECT_H
