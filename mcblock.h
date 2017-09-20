#ifndef MCBLOCK_H
#define MCBLOCK_H

#include "geometricobject.h"

enum BlockID {
    Stone =     1,
    GrassSide = 2,
    Dirt =      3,
    WaterFlow = 8,
    WaterStill = 9,
    Sand =      12,
    LogOak =    17,
    LeavesOak = 18,
    FarmLand =  60,

};

class MCBlock : public GeometricObject
{
public:
    MCBlock();
    bool air;

    Material* mat_top;
    Material* mat_side;
    BlockID _id;

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &tmin) const;
};

#endif // MCBLOCK_H
