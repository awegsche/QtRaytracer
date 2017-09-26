#ifndef GLOSSYSPECULAR_H
#define GLOSSYSPECULAR_H

#include "brdf.h"
#include "constantcolor.h"
#include "texture.h"


class GlossySpecular : public BRDF
{
private:
    Texture* ks;
    real exp;
    Texture* cs;

public:
    GlossySpecular();
    GlossySpecular(real kspecular, real exponent);
    GlossySpecular(real kspecular, real exponent, const RGBColor& color);

    void set_color(Texture *t);
    void set_k(const real k);
    void set_k(Texture* t);
    void set_exp(const real e);

    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;
};

#endif // GLOSSYSPECULAR_H
