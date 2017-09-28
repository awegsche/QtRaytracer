#include "normal.h"
#include "vector.h"
#include <math.h>

Normal::Normal()
{

}

Normal::Normal(real x, real y, real z)
    : data(x,y,z, 0.0){

}

Normal::Normal(const Vector &v)
    : data(v.X(), v.Y(), v.Z(), 0.0) {

}

Normal::Normal(const Normal &n)
    : data(n.X(), n.Y(), n.Z(), 0.0) {

}

Normal::Normal(const real4 & xyzw)
	:data(xyzw)
{
}

void Normal::normalize()
{
    using namespace std;
    real over_length = 1.0 / sqrt(add_horizontal(data * data));
	data = data * real4(over_length,over_length, over_length, 1.0);
}


real Normal::operator*(const Vector &u)
{
    return add_horizontal(data * u.data);
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
    return Normal(-data[3], -data[2], -data[1]);
}

real operator*(const Vector &u, const Normal &n)
{
    return add_horizontal(u.data * n.data);
}

Vector operator *(real a, const Normal &n)
{
    return Vector(n.data * real4(a, a, a, a));
}

Vector operator +(const Normal &n, const Normal &m)
{
    return Vector(n.data + m.data);
}

real operator*(const Normal &n, const Vector &u)
{
    return add_horizontal(n.data * u.data);
}

Vector operator *(const Normal &n, real a)
{
    return Vector(n.data * real4(a,a,a,a));
}
