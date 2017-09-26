#include "bbox.h"

BBox::BBox()
    :   x0(0.0), y0(0.0), z0(0.0),
        x1(0.0), y1(0.0), z1(0.0)
{

}

BBox::BBox(Point p0, Point p1)
    :   x0(p0.X), y0(p0.Y), z0(p0.Z),
        x1(p1.X), y1(p1.Y), z1(p1.Z)

{

}

BBox::BBox(real x0_, real y0_, real z0_, real x1_, real y1_, real z1_)
    :   x0(x0_), y0(y0_), z0(z0_),
        x1(x1_), y1(y1_), z1(z1_) {

}

bool BBox::inside(const Point &p) const
{
     return ((p.X > x0 && p.X < x1) && (p.Y > y0 && p.Y < y1) && (p.Z > z0 && p.Z < z1));
}

bool BBox::hit(const Ray &ray) const
{
    real ox = ray.o.X; real oy = ray.o.Y; real oz = ray.o.Z;
    real dx = ray.d.X; real dy = ray.d.Y; real dz = ray.d.Z;

    real tx_min, ty_min, tz_min;
    real tx_max, ty_max, tz_max;

    real a = 1.0/dx;
    if ( a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    real b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    real c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    real t0, t1;

    if (tx_min > ty_min)
        t0 = tx_min;
    else
        t0 = ty_min;
    if (tz_min > t0)
        t0 = tz_min;

    if (tx_max < ty_max)
        t1 = tx_max;
    else
        t1 = ty_max;
    if (tz_max < t0)
        t1 = tz_max;

    return t0 < t1 && t1 > kEpsilon;
}
