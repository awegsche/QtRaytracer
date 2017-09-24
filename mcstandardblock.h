#ifndef MCSTANDARDBLOCK_H
#define MCSTANDARDBLOCK_H

#include "mcblock.h"

class MCStandardBlock : public MCBlock
{
public:
    MCStandardBlock();
    MCStandardBlock(Material* material);

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &tmin) const;

    // MCBlock interface
public:
    bool block_hit(const Ray &ray, const Point &p0, real &tmin, ShadeRec &sr) const;
};

#endif // MCSTANDARDBLOCK_H
