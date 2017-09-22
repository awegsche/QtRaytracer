#ifndef BRDF_H
#define BRDF_H

#include "rgbcolor.h"
#include "shaderec.h"
#include "vector.h"

class Sampler;


class BRDF
{
protected:
    Sampler* sampler_ptr;
public:
    BRDF();

    virtual RGBColor f(const ShadeRec& sr, const Vector& wi, const Vector& wo) const;
    virtual RGBColor sample_f(const ShadeRec& sr, Vector& wi, const Vector& wo) const;
    virtual RGBColor rho(const ShadeRec& sr, const Vector& wo) const;

    virtual real transparency(const ShadeRec& sr);
};

#endif // BRDF_H
