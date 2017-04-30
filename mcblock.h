#ifndef MCBLOCK_H
#define MCBLOCK_H

#include "geometricobject.h"

class MCBlock : public GeometricObject
{
public:
    MCBlock();
    bool air;

    Material* mat_top;
    Material* mat_side;

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &tmin) const;
};

#endif // MCBLOCK_H
