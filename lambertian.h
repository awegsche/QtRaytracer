#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "brdf.h"

class Lambertian : public BRDF
{
private:
    float kd;
    RGBColor cd;

public:
    Lambertian();

    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;
};

#endif // LAMBERTIAN_H
