#include "glossyspecular.h"

#include "rgbcolor.h"
#include "vector.h"
#include "myexception.h"


GlossySpecular::GlossySpecular()
    : ks(1.0), exp(1.0), cs(1.0)
{

}

GlossySpecular::GlossySpecular(real kspecular, real exponent)
    : ks(kspecular), exp(exponent), cs(1.0)
{

}

GlossySpecular::GlossySpecular(real kspecular, real exponent, const RGBColor &color)
    : ks(kspecular), exp(exponent), cs(color)
{

}

RGBColor GlossySpecular::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    RGBColor L;
    real ndotwi = sr.normal * wi;
    Vector r(-wi + 2.0 * sr.normal * ndotwi);
    real rdotwi = r * wi;

    if (rdotwi > .0)
        L = ks * pow(rdotwi, exp);

    return L;
}

RGBColor GlossySpecular::sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    MyException E("sample_f not yet implemented.");
    E.raise();
}

RGBColor GlossySpecular::rho(const ShadeRec &sr, const Vector &wo) const
{
    return RGBColor(.0);
}
