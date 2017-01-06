#ifndef CONCURRENTSTRUCT_H
#define CONCURRENTSTRUCT_H

#include "ray.h"
#include "point2d.h"
#include "viewplane.h"
#include "world.h"

class ConcurrentStruct
{
public:
    Ray ray;
    Point2D pp;
    ViewPlane vp;
    Point2D sp;
    int depth;
    RGBColor L;
    World *w;
    Point2D pixel_point;
public:
    ConcurrentStruct();
    ConcurrentStruct(Ray& r_, Point2D& pp_, ViewPlane& vp_, Point2D& sp_,
                     int depth_, RGBColor& l_,
                     World* w_, Point2D& pix_point_ );
};

#endif // CONCURRENTSTRUCT_H
