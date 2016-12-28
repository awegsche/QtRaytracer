#ifndef PLANE_H
#define PLANE_H

#include "point.h"
#include "normal.h"
#include "constants.h"
#include "geometricobject.h"

class Plane : public GeometricObject
{
public:
    Point point;
    Normal normal;
public:
    Plane();

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
};

#endif // PLANE_H
