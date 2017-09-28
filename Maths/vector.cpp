#include "vector.h"
#include "math.h"

Vector::Vector()
    :data(0.0, 0.0, 0.0, 0.0)
{
}

Vector::Vector(real a)
    : data(a, a, a, 0.0) {

}

Vector::Vector(real x, real y, real z)
    :data(x, y, z, 0.0)
{

}

Vector::Vector(const real4 & xyzw)
	:data(xyzw)
{
}

Vector::Vector(const Vector &v)
    : data(v.data)
{

}

void Vector::normalize()
{
    real l = 1.0 / length();
	data *= real4(l, l, l, l);
}
real Vector::length() const

{
    return sqrt(add_horizontal(data * data));
}

Vector &Vector::operator=(const Vector &v)
{
	data = v.data;

    return *this;
}

Vector &Vector::operator=(const Normal &n)
{
	data = n.data;

    return *this;
}

Vector Vector::operator-() const
{
    return Vector(data * real4(-1.0, -1.0, -1.0, -1.0));
}

Vector Vector::operator+=(const Vector &v)
{
	data += v.data;

    return *this;
}

Vector Vector::hat() const
{
    real one_over_l = 1.0 / this->length();
    return Vector(data * real4(one_over_l, one_over_l, one_over_l, one_over_l));
}


const Vector operator*(const Vector &v, real a)
{
    return Vector(v.data * real4(a,a,a,a));
}

const Vector operator*(real a, const Vector &v)
{
    return Vector(v.data *  real4(a, a, a, a));
}

real operator *(const Vector &a, const Vector &b)
{
    return add_horizontal(a.data * b.data);
}

const Vector operator+(const Vector &a, const Vector &b)
{
    return Vector(a.data + b.data);
}

const Vector operator/(const Vector &v, real a)
{
    real  b = 1.0/a;
    return Vector(v.data *  real4(b, b, b, b));
}

const Vector operator^(const Vector &a, const Vector &b)
{
    return Vector(a.Y() * b.Z() - a.Z() * b.Y(), a.Z() * b.X() - a.X() * b.Z(), a.X() * b.Y() - a.Y() * b.X());
}

const Vector operator-(const Vector &a, const Vector &b)
{
    return Vector(a.data - b.data);
}
