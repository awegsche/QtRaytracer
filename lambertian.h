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


    void set_k(float k);
    void set_color(const RGBColor& color);

    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;

    // BRDF interface
public:
    RGBColor sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
};

#endif // LAMBERTIAN_H
