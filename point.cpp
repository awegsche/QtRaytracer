#include "point.h"
#include "vector.h"

Point::Point()
{

}

Point::Point(real x, real y)
    : X(x), Y(y)
{

}

Point Point::operator+(const Point &p, const Vector &v)
{
    return Point(p.X + v.X, p.Y + v.Y);
}
