#ifndef NORMAL_H
#define NORMAL_H

#include "constants.h"
class Vector;

class Normal
{
public:
    real X;
    real Y;
    real Z;
public:
    Normal();
    Normal(real x, real y, real z);
    Normal(const Vector& v);
    Normal(const Normal& n);

    void normalize();
    //real operator* (const Vector& u, const Normal& n);
    real operator* (const Vector& u);
    Normal& operator=(const Vector& v);
    Normal &operator+=(const Normal& n);

    Normal operator-();
};

real operator*(const Vector& u, const Normal& n);
real operator*(const Normal& n, const Vector& v);
Vector operator *(real a, const Normal& n);
Vector operator *(const Normal& n, real a);
Vector operator +( const Normal& n, const Normal& m);
#endif // NORMAL_H
