#ifndef GEOMETRICOBJECT_H
#define GEOMETRICOBJECT_H

#include "constants.h"

class Ray;
class ShadeRec;

class GeometricObject
{
public:
    GeometricObject();

    virtual bool hit(const Ray& ray, real& tmin, ShadeRec &sr);


};

#endif // GEOMETRICOBJECT_H
