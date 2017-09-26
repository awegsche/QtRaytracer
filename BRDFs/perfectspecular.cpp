#include "perfectspecular.h"
#include "constantcolor.h"

PerfectSpecular::PerfectSpecular()
    :kr(0), cr()
{

}

PerfectSpecular::PerfectSpecular(const real cr_red, const real cr_green, const real cr_blue, const real _kr)
{
    kr = new ConstantColor(RGBColor(_kr, _kr, _kr));
    cr = new ConstantColor(RGBColor(cr_red, cr_green, cr_blue));
}

void PerfectSpecular::set_color(const real r, const real g, const real b)
{
    cr = new ConstantColor(RGBColor(r, g, b));
}

void PerfectSpecular::set_kr(const real k)
{
    kr = new ConstantColor(k, k, k);
}

void PerfectSpecular::set_kr(Texture *t)
{
    kr = t;
}

RGBColor PerfectSpecular::sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const
{
    float ndotwo = sr.normal * wo;
    wi = -wo + 2.0 * sr.normal * ndotwo;

    return (kr->get_color(sr).b * cr->get_color(sr) / (sr.normal * wi));
}
