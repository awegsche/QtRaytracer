#include "vector.h"
#include "math.h"

Vector::Vector()
    :data(0.0, 0.0, 0.0)
{
}

Vector::Vector(real a)
    : data(a, a, a) {

}

Vector::Vector(real x, real y, real z)
    :data(x, y, z)
{

}

Vector::Vector(const real3 &xyz)
    :data(xyz)
{
}

Vector::Vector(const Vector &v)
    : data(v.data)
{

}

void Vector::normalize()
{
    real l = 1.0 / length();
    data *= l;
}
real Vector::length() const

{
    return vector_length(data);
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
    return Vector(-data);
}

Vector Vector::operator+=(const Vector &v)
{
	data += v.data;

    return *this;
}

Vector Vector::hat() const
{
    real one_over_l = 1.0 / this->length();
    return Vector(data * one_over_l);
}


const Vector operator*(const Vector &v, real a)
{
    return Vector(v.data * a);
}

const Vector operator*(real a, const Vector &v)
{
    return Vector(v.data *  a);
}

real operator *(const Vector &a, const Vector &b)
{
    return dot_product(a.data, b.data);
}

const Vector operator+(const Vector &a, const Vector &b)
{
    return Vector(a.data + b.data);
}

const Vector operator/(const Vector &v, real a)
{
    real  b = 1.0/a;
    return Vector(v.data *  b);
}

const Vector operator^(const Vector &a, const Vector &b)
{
    return Vector(cross_product(a.data, b.data));
}

const Vector operator-(const Vector &a, const Vector &b)
{
    return Vector(a.data - b.data);
}
