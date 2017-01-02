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

    void normalize();
    //real operator* (const Vector& u, const Normal& n);
    real operator* (const Vector& u);
    Normal& operator=(const Vector& v);
    Normal &operator+=(const Normal& n);

    Normal operator-();
};

real operator*(const Vector& u, const Normal& n);
Normal operator *(real a, const Normal& n);
Normal operator +( const Normal& n, const Normal& m);
#endif // NORMAL_H
