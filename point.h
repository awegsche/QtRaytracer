#ifndef POINT_H
#define POINT_H
#include "constants.h"

class Vector;

class Point
{
public:
    real X;
    real Y;
    real Z;
public:
    Point();
    Point(real x, real y, real z);

};

const Point operator+(const Point& p, const Vector& v);
const Vector operator-(const Point& a, const Point& b);

#endif // POINT_H
