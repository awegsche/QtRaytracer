#ifndef MCWATERBLOCK_H
#define MCWATERBLOCK_H

#include "mcblock.h"
#include "mcstandardblock.h"

class MCWaterBlock : public MCStandardBlock
{
public:
    MCWaterBlock();
    MCWaterBlock(Material* material);

    // GeometricObject interface
public:
    bool shadow_hit(const Ray &ray, real &tmin) const Q_DECL_OVERRIDE;

    // MCBlock interface
public:
    bool block_hit(const Ray &ray, const Point &p0, real &tmin, ShadeRec &sr) const Q_DECL_OVERRIDE;
};

#endif // MCWATERBLOCK_H
