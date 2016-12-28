#ifndef SHADEREC_H
#define SHADEREC_H

#include "constants.h"
#include "point.h"
#include "normal.h"
#include "rgbcolor.h"
#include "world.h"


class ShadeRec
{
public:
    bool hit_an_object;
    Point  local_hit_point;
    Normal normal;
    RGBColor color;
    World *w;


    ShadeRec(World *world);
};

#endif // SHADEREC_H
