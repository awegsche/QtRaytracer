#include "lambertian.h"
#include "constantcolor.h"

Lambertian::Lambertian()
{

}

Lambertian::Lambertian(float k, const RGBColor &color)
    : kd(k){
    cd = new ConstantColor(color);
}


void Lambertian::set_k(float k)
{
    kd = k;
}

void Lambertian::set_color(const RGBColor &color)
{
    cd = new ConstantColor(color);
}

void Lambertian::set_color(Texture *t)
{
    cd = t;
}


RGBColor Lambertian::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return (kd * cd->get_color(sr) * invPi);
}

RGBColor Lambertian::rho(const ShadeRec &sr, const Vector &wo) const
{
    return kd * cd->get_color(sr);
}

RGBColor Lambertian::sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return RGBColor();
}

