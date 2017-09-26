#ifndef BBOX_H
#define BBOX_H

#include "constants.h"
#include "ray.h"
#include "point.h"

class BBox
{
public:
    real x0, y0, z0, x1, y1, z1;
public:
    BBox();
    BBox(Point p0, Point p1);
    BBox(real x0_, real y0_, real z0_, real x1_, real y1_, real z1_);

    bool inside(const Point& p) const;

    bool hit(const Ray& ray) const;
};

#endif // BBOX_H
