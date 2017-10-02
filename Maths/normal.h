#ifndef NORMAL_H
#define NORMAL_H

#include "constants.h"
class Vector;

class Normal
{
public:
    real3 data;
public:
    Normal();
    Normal(real x, real y, real z);
    Normal(const Vector& v);
    Normal(const Normal& n);
    Normal(const real3& xyz);

    void normalize();
    //real operator* (const Vector& u, const Normal& n);
    real operator* (const Vector& u);
    Normal& operator=(const Vector& v);
    Normal &operator+=(const Normal& n);

    Normal operator-();

	real X() const {
        return data.get_x();
	}
	real Y() const {
        return data.get_y();
	}
	real Z() const {
        return data.get_z();
	}
};

real operator*(const Vector& u, const Normal& n);
real operator*(const Normal& n, const Vector& v);
Vector operator *(real a, const Normal& n);
Vector operator *(const Normal& n, real a);
Vector operator +( const Normal& n, const Normal& m);
#endif // NORMAL_H
