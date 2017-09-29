#include "normal.h"
#include "vector.h"
#include <math.h>

Normal::Normal()
{

}

Normal::Normal(real x, real y, real z)
    : data(x,y,z){

}

Normal::Normal(const Vector &v)
    : data(v.X(), v.Y(), v.Z()) {

}

Normal::Normal(const Normal &n)
    : data(n.X(), n.Y(), n.Z()) {

}

Normal::Normal(const real3 &xyz)
    :data(xyz)
{
}

void Normal::normalize()
{

    data = normalize_vector(data);
}


real Normal::operator*(const Vector &u)
{
    return dot_product(data, u.data);
}

Normal &Normal::operator=(const Vector &v)
{
	data = v.data;

    return *this;
}

Normal &Normal::operator+=(const Normal &n)
{
	data += n.data;

    return *this;
}

Normal Normal::operator-()
{
    return Normal(-data);
}

real operator*(const Vector &u, const Normal &n)
{
    return dot_product(u.data, n.data);
}

Vector operator *(real a, const Normal &n)
{
    return Vector(n.data * a);
}

Vector operator +(const Normal &n, const Normal &m)
{
    return Vector(n.data + m.data);
}

real operator*(const Normal &n, const Vector &u)
{
    return dot_product(n.data, u.data);
}

Vector operator *(const Normal &n, real a)
{
    return Vector(n.data * a);
}
