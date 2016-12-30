#include "lambertian.h"

Lambertian::Lambertian()
{

}

RGBColor Lambertian::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return (kd * cd * invPi);
}

RGBColor Lambertian::rho(const ShadeRec &sr, const Vector &wo) const
{
    return kd * cd;
}
