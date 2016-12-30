#ifndef SHADEREC_H
#define SHADEREC_H

#include "constants.h"
#include "point.h"
#include "normal.h"
#include "rgbcolor.h"
#include "world.h"
#include "material.h"

class Ray;
class Vector;


class ShadeRec
{
public:
    bool hit_an_object;
    Point  local_hit_point;     // texture coordinates
    Point hitPoint;             // world coordinates of hit point
    Material* material_ptr;     // material of nearest hit object
    Normal normal;
    RGBColor color;   //------------ delete !
    Ray ray;                    // for speculiar highlights
    int depth;
    Vector dir;                 // for area lights
    real t;
    World *w;


    ShadeRec(World *world);
    ShadeRec(const ShadeRec& sr);
};

#endif // SHADEREC_H
