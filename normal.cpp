#include "normal.h"
#include "vector.h"
#include <cmath>

Normal::Normal()
{

}

Normal::Normal(real x, real y, real z)
    : X(x), Y(y), Z(z){

}

void Normal::normalize()
{
    using namespace std;
    real over_length = 1.0 / sqrt(X * X + Y * Y + Z * Z);
    X *= over_length;
    Y *= over_length;
    Z *= over_length;
}


real Normal::operator*(const Vector &u)
{
    return this->X * u.X + this->Y * u.Y + this->Z * u.Z;
}

Normal &Normal::operator=(const Vector &v)
{
    this->X = v.X;
    this->Y = v.Y;
    this->Z = v.Z;

    return *this;
}

Normal &Normal::operator+=(const Normal &n)
{
    X += n.X;
    Y += n.Y;
    Z += n.Z;

    return *this;
}

Normal Normal::operator-()
{
    return Normal(-X, -Y, -Z);
}

real operator*(const Vector &u, const Normal &n)
{
    return n.X * u.X + n.Y * u.Y + n.Z * u.Z;
}

Normal operator *(real a, const Normal &n)
{
    return Normal(a * n.X, a * n.Y, a * n.Z);
}

Normal operator +(const Normal &n, const Normal &m)
{
    return Normal(m.X + n.X, m.Y + n.Y, m.Z + n.Z);
}
