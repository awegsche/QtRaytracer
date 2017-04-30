#include "point.h"
#include "vector.h"

Point::Point()
{

}

Point::Point(real x, real y, real z)
    : X(x), Y(y), Z(z)
{

}

const Point &Point::operator=(const Vector &v)
{
    X = v.X;
    Y = v.Y;
    Z = v.Z;

    return *this;
}

Point &Point::operator +=(const Vector &v)
{
    X += v.X;
    Y += v.Y;
    Z += v.Z;

    return *this;
}

const Point operator+(const Point &p, const Vector &v)
{
    return Point(p.X + v.X, p.Y + v.Y, p.Z + v.Z);
}

const Vector operator-(const Point &a, const Point &b)
{
    return Vector(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
}

const Point operator-(const Point &p, const Vector &v)
{
     return Point(p.X - v.X, p.Y - v.Y, p.Z - v.Z);
}
