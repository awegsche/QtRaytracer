#include "point.h"
#include "vector.h"

Point::Point()
	:data(0.0,0.0,0.0,0.0)
{

}

Point::Point(real x, real y, real z)
	: data(x, y, z, 0.0)
{

}

Point::Point(const real4 & xyzw)
	:data(xyzw)
{
}

const Point &Point::operator=(const Vector &v)
{
	data = v.data;
    return *this;
}

Point &Point::operator +=(const Vector &v)
{
	data += v.data;

    return *this;
}

const Point operator+(const Point &p, const Vector &v)
{
    return Point(p.data + v.data);
}

const Vector operator-(const Point &a, const Point &b)
{
    return Vector(a.data - b.data);
}

const Point operator-(const Point &p, const Vector &v)
{
     return Point(p.data - v.data);
}
