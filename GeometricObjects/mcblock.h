#ifndef MCBLOCK_H
#define MCBLOCK_H

#include "geometricobject.h"

enum BlockID {
    Stone =     1,
    GrassSide = 2,
    Dirt =      3,
    CobbleStone = 4,
    OakWoodPlank = 5,
    WaterFlow = 8,
    WaterStill = 9,
    Sand =      12,
    LogOak =    17,
    LeavesOak = 18,
    Grass = 31,
    Dandelion = 37,
    Poppy = 38,
    FarmLand =  60,
    SugarCanes = 83
};

class MCBlock : public GeometricObject
{
public:
    MCBlock();

    // GeometricObject interface
public:
//    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const = 0;
    virtual bool block_hit(const Ray &ray, const Point& p0, real &tmin, ShadeRec &sr) const = 0;
//    bool shadow_hit(const Ray &ray, real &tmin) const = 0;

};

#endif // MCBLOCK_H
