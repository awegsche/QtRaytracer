#ifndef PHONG_H
#define PHONG_H

#include "material.h"
#include "lambertian.h"
#include "glossyspecular.h"


class Phong : public Material
{
protected:
    Lambertian* ambient_brdf;
    Lambertian* diffuse_brdf;
    GlossySpecular* specular_brdf;

public:
    Phong();
    Phong(real kambient, real kdiffuse, real kspecular, real expspecular, float r, float g, float b);

    void set_color(const real r, const real g, const real b);
    void set_ambient_color(const real r, const real g, const real b);
    void set_diffuse_color(const real r, const real g, const real b);
    void set_specular_color(const real r, const real g, const real b);

    void set_ambient_color(Texture* t);
    void set_diffuse_color(Texture* t);
    void set_specular_color(Texture* t);

    void set_ka(const real k);
    void set_kd(const real k);
    void set_ks(const real k);
    void set_ks(Texture* t);

    void set_exp(const real e);


    // Material interface
public:
    RGBColor shade(ShadeRec &sr);

    // Material interface
public:
    RGBColor noshade(ShadeRec &sr) Q_DECL_OVERRIDE;
};

#endif // PHONG_H
