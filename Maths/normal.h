#ifndef NORMAL_H
#define NORMAL_H

#include "constants.h"
class Vector;

class Normal
{
public:
	real4 data;
public:
    Normal();
    Normal(real x, real y, real z);
    Normal(const Vector& v);
    Normal(const Normal& n);
	Normal(const real4& xyzw);

    void normalize();
    //real operator* (const Vector& u, const Normal& n);
    real operator* (const Vector& u);
    Normal& operator=(const Vector& v);
    Normal &operator+=(const Normal& n);

    Normal operator-();

	const real X() const {
		return data[3];
	}
	const real Y() const {
		return data[2];
	}
	const real Z() const {
		return data[1];
	}
};

real operator*(const Vector& u, const Normal& n);
real operator*(const Normal& n, const Vector& v);
Vector operator *(real a, const Normal& n);
Vector operator *(const Normal& n, real a);
Vector operator +( const Normal& n, const Normal& m);
#endif // NORMAL_H
