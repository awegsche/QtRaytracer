#ifndef SHADEREC_H
#define SHADEREC_H

#include "constants.h"

class Point;
class Vector;
class Normal;
class RGBColor;
class World;


class ShadeRec
{
public:
    bool hit_an_object;
    Point  local_hit_point;
    Normal normal;
    RGBColor color;
    World& w;


    ShadeRec(World& world);
};

#endif // SHADEREC_H
