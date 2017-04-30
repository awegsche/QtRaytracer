#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "brdf.h"
#include "texture.h"

class Lambertian : public BRDF
{
private:
    float kd;
    Texture* cd;

public:
    Lambertian();
    Lambertian(float k, const RGBColor& color);


    void set_k(float k);
    void set_color(const RGBColor& color);
    void set_color(Texture* t);

    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;

    // BRDF interface
public:
    RGBColor sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
};

#endif // LAMBERTIAN_H
