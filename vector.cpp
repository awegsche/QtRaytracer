#include "vector.h"

Vector::Vector()
{

}

Vector::Vector(real x, real y, real z)
    : X(x), Y(y), Z(z)
{

}


const Vector operator*(const Vector &v, real a)
{
    return Vector(a * v.X, a * v.Y, a * v.Z);
}

const Vector operator*(real a, const Vector &v)
{
    return Vector(a * v.X, a * v.Y, a * v.Z);
}

const real operator *(const Vector &a, const Vector &b)
{
    return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
}

const Vector operator+(const Vector &a, const Vector &b)
{
    return Vector(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
}

const Vector operator/(const Vector &v, real a)
{
    real  b = 1.0/a;
    return Vector(b * v.X, b * v.Y, b * v.Z);
}
