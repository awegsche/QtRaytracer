#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "brdf.h"
#include "texture.h"

class Lambertian : public BRDF
{
public:
    real kd;
    Texture* cd;

public:
    Lambertian();
    Lambertian(real k, const RGBColor& color);


    void set_k(real k);
    void set_color(const RGBColor& color);
    void set_color(Texture* t);


    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;

    // BRDF interface
public:
    RGBColor sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const;

    // BRDF interface
public:
    real transparency(const ShadeRec &sr) Q_DECL_OVERRIDE;
};

#endif // LAMBERTIAN_H
