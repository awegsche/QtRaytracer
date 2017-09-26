#ifndef SPHERE_H
#define SPHERE_H

#include "geometricobject.h"
#include "point.h"
#include "constants.h"
#include "rgbcolor.h"

class Sphere : public GeometricObject
{
public:
    Point m;
    real r;
public:
    Sphere();
    Sphere(Point center, real radius);

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;

    // GeometricObject interface
public:
    BBox get_bounding_box();

    // GeometricObject interface
public:
    bool shadow_hit(const Ray &ray, real &tmin) const;
};

#endif // SPHERE_H
