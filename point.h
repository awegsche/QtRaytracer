#ifndef POINT_H
#define POINT_H
#include "constants.h"

class Vector;

class Point
{
public:
    real X;
    real Y;
public:
    Point();
    Point(real x, real y);

    Point operator+ (const Point& p, const Vector& v);

};

#endif // POINT_H
