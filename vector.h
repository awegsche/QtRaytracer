#ifndef VECTOR_H
#define VECTOR_H

#include "constants.h"
#include "normal.h"

class Vector
{
public:
    real X;
    real Y;
    real Z;
public:
    Vector();
    Vector(real x, real y, real z);

    void normalize();
    real length();

    const Vector& operator=(const Vector& v);

    Vector hat();
};

const Vector operator*(const Vector& v, real a);
const Vector operator*(real a, const Vector& v);
const Vector operator/(const Vector& v, real a);
const real operator *(const Vector& a, const Vector& b);
const Vector operator^(const Vector& a, const Vector& b);


const Vector operator+(const Vector& a, const Vector& b);
const Vector operator-(const Vector& a, const Vector& b);


#endif // VECTOR_H
