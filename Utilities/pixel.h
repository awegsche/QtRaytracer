#ifndef PIXEL_H
#define PIXEL_H

#include "rgbcolor.h"
#include "point2d.h"
class World;

class Pixel
{
public:
    RGBColor color;
    Point2D point;
    World* w;
public:
    Pixel();
    Pixel(const RGBColor& color_, const Point2D& p, World* w_);
};

#endif // PIXEL_H
