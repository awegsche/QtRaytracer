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

    //real operator* (const Vector& u, const Normal& n);
    real operator* (const Vector& u);
    const Normal& operator=(const Vector& v);
};

const real operator*(const Vector& u, const Normal& n);
#endif // NORMAL_H
