#include "normal.h"
#include "vector.h"
#include <cmath>

Normal::Normal()
{

}

Normal::Normal(real x, real y, real z)
    : X(x), Y(y), Z(z){

}

Normal::Normal(const Vector &v)
    : X(v.X), Y(v.Y), Z(v.Z) {

}

Normal::Normal(const Normal &n)
    : X(n.X), Y(n.Y), Z(n.Z) {

}

void Normal::normalize()
{
    using namespace std;
    real over_length = 1.0 / sqrt(X * X + Y * Y + Z * Z);
    X *= over_length;
    Y *= over_length;
    Z *= over_length;
}


real Normal::operator*(const Vector &u)
{
    return this->X * u.X + this->Y * u.Y + this->Z * u.Z;
}

Normal &Normal::operator=(const Vector &v)
{
    this->X = v.X;
    this->Y = v.Y;
    this->Z = v.Z;

    return *this;
}

Normal &Normal::operator+=(const Normal &n)
{
    X += n.X;
    Y += n.Y;
    Z += n.Z;

    return *this;
}

Normal Normal::operator-()
{
    return Normal(-X, -Y, -Z);
}

real operator*(const Vector &u, const Normal &n)
{
    return n.X * u.X + n.Y * u.Y + n.Z * u.Z;
}

Vector operator *(real a, const Normal &n)
{
    return Vector(a * n.X, a * n.Y, a * n.Z);
}

Vector operator +(const Normal &n, const Normal &m)
{
    return Vector(m.X + n.X, m.Y + n.Y, m.Z + n.Z);
}

real operator*(const Normal &n, const Vector &u)
{
    return n.X * u.X + n.Y * u.Y + n.Z * u.Z;
}

Vector operator *(const Normal &n, real a)
{
    return Vector(n.X * a, n.Y * a, n.Z * a);
}
