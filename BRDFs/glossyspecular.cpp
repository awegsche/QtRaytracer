#include "glossyspecular.h"

#include "rgbcolor.h"
#include "vector.h"
#include "myexception.h"


GlossySpecular::GlossySpecular()
    : ks(new ConstantColor(1.0, 1.0, 1.0)), exp(1.0)
{
    cs = new ConstantColor(RGBColor((real)1.0));
}

GlossySpecular::GlossySpecular(real kspecular, real exponent)
    : ks(new ConstantColor(kspecular, kspecular, kspecular)), exp(exponent)
{
     cs = new ConstantColor(RGBColor((real)1.0));
}

GlossySpecular::GlossySpecular(real kspecular, real exponent, const RGBColor &color)
    : ks(new ConstantColor(kspecular, kspecular, kspecular)), exp(exponent)
{
    cs = new ConstantColor(color);
}

void GlossySpecular::set_color(Texture *t)
{
    cs = t;
}

void GlossySpecular::set_k(const real k)
{
    ks = new ConstantColor(k, k, k);
}

void GlossySpecular::set_k(Texture *t)
{
    ks = t;
}

void GlossySpecular::set_exp(const real e)
{
    exp = e;
}

RGBColor GlossySpecular::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    RGBColor L;
    real ndotwi = sr.normal * wi;
    Vector r(-wi + (2.0 * sr.normal * ndotwi));
    //r = r.hat();        // very experimental
    real rdotwo = r * wo;

    if (rdotwo > .0)
        L = ks->get_color(sr).r * (real)pow(rdotwo, exp) * cs->get_color(sr);

    return L;
}

RGBColor GlossySpecular::sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const
{
    MyException E("sample_f not yet implemented.");
    E.raise();
    return RGBColor();
}

RGBColor GlossySpecular::rho(const ShadeRec &sr, const Vector &wo) const
{
    return RGBColor((real).0);
}
