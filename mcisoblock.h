#ifndef MCISOBLOCK_H
#define MCISOBLOCK_H

#include "mcblock.h"

class MCIsoBlock : public MCBlock
{
private:
    Material* top_mat;
    Material* bottom_mat;
public:
    MCIsoBlock();
    MCIsoBlock(Material* top, Material* sides);
    MCIsoBlock(Material* top, Material* bottom, Material* sides);

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &tmin) const;

    // MCBlock interface
public:
    bool block_hit(const Ray &ray, const Point &p0, real &tmin, ShadeRec &sr) const;
};

#endif // MCISOBLOCK_H
