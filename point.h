#ifndef POINT_H
#define POINT_H
#include "constants.h"

class Point
{
public:
    real x;
    real y;
public:
    Point();

    Point(real x, real y);

    Point operator+ (const Point& p, const Vector& v);

};

#endif // POINT_H
