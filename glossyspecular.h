#ifndef GLOSSYSPECULAR_H
#define GLOSSYSPECULAR_H

#include "brdf.h"
#include "constantcolor.h"
#include "texture.h"


class GlossySpecular : public BRDF
{
private:
    real ks;
    real exp;
    Texture* cs;

public:
    GlossySpecular();
    GlossySpecular(real kspecular, real exponent);
    GlossySpecular(real kspecular, real exponent, const RGBColor& color);

    void set_color(Texture *t);

    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;
};

#endif // GLOSSYSPECULAR_H
