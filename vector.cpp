#include "vector.h"
#include "cmath"

Vector::Vector()
    :X(0.0), Y(0.0), Z(0.0){

}

Vector::Vector(real a)
    : X(a), Y(a), Z(a) {

}

Vector::Vector(real x, real y, real z)
    : X(x), Y(y), Z(z)
{

}

void Vector::normalize()
{
    real l = 1.0 / length();
    X *= l;
    Y *= l;
    Z *= l;
}
real Vector::length() const

{
    return sqrt(X * X + Y * Y + Z * Z);
}

const Vector &Vector::operator=(const Vector &v)
{
    this->X = v.X;
    this->Y = v.Y;
    this->Z = v.Z;

    return *this;
}

const Vector Vector::operator-()
{
    return Vector(-X, -Y, -Z);
}

Vector Vector::hat() const
{
    real one_over_l = 1.0 / this->length();
    return Vector(X * one_over_l, Y * one_over_l, Z * one_over_l);
}


const Vector operator*(const Vector &v, real a)
{
    return Vector(a * v.X, a * v.Y, a * v.Z);
}

const Vector operator*(real a, const Vector &v)
{
    return Vector(a * v.X, a * v.Y, a * v.Z);
}

real operator *(const Vector &a, const Vector &b)
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

const Vector operator^(const Vector &a, const Vector &b)
{
    return Vector(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
}

const Vector operator-(const Vector &a, const Vector &b)
{
    return Vector(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
}
