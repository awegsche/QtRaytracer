#include "lambertian.h"

Lambertian::Lambertian()
{

}

void Lambertian::set_k(float k)
{
    kd = k;
}

void Lambertian::set_color(const RGBColor &color)
{
    cd = color;
}

RGBColor Lambertian::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return (kd * cd * invPi);
}

RGBColor Lambertian::rho(const ShadeRec &sr, const Vector &wo) const
{
    return kd * cd;
}

RGBColor Lambertian::sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return RGBColor();
}
