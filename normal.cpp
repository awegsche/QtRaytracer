#include "normal.h"
#include "vector.h"
Normal::Normal()
{

}

Normal::Normal(real x, real y, real z)
    : X(x), Y(y), Z(z){

}


real Normal::operator*(const Vector &u)
{
    return this->X * u.X + this->Y * u.Y + this->Z * u.Z;
}

const Normal &Normal::operator=(const Vector &v)
{
    this->X = v.X;
    this->Y = v.Y;
    this->Z = v.Z;

    return *this;
}

const real operator*(const Vector &u, const Normal &n)
{
    return n.X * u.X + n.Y * u.Y + n.Z * u.Z;
}
