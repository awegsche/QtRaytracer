#ifndef VECTOR_H
#define VECTOR_H

#include "constants.h"
#include "normal.h"


class Vector
{
public:
    real3 data;
public:
    Vector();
    Vector(real a);
    Vector(real x, real y, real z);
    Vector(const real3& xyz);
    Vector(const Vector& v);

    void normalize();
    real length() const;

    Vector& operator=(const Vector& v);
    Vector& operator=(const Normal& n);
    Vector operator-() const;
    Vector operator+=(const Vector& v);

    Vector hat() const;

	real X() const{
        return data.get_x();
	}
	real Y() const{
        return data.get_y();
	}
	real Z() const{
        return data.get_z();
	}

};

const Vector operator*(const Vector& v, real a);
const Vector operator*(real a, const Vector& v);
const Vector operator/(const Vector& v, real a);
real operator *(const Vector& a, const Vector& b);
const Vector operator^(const Vector& a, const Vector& b);


const Vector operator+(const Vector& a, const Vector& b);
const Vector operator-(const Vector& a, const Vector& b);


#endif // VECTOR_H
