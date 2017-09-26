#include "brdf.h"

BRDF::BRDF()
{

}

RGBColor BRDF::f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const
{
    return RGBColor();
}

RGBColor BRDF::sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const
{
    return RGBColor();

}

RGBColor BRDF::rho(const ShadeRec &sr, const Vector &wo) const
{
    return RGBColor();

}

real BRDF::transparency(const ShadeRec &sr)
{
    return 1.0;
}
