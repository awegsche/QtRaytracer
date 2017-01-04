#ifndef GLOSSYSPECULAR_H
#define GLOSSYSPECULAR_H

#include "brdf.h"



class GlossySpecular : public BRDF
{
private:
    real ks;
    real exp;
    RGBColor cs;

public:
    GlossySpecular();
    GlossySpecular(real kspecular, real exponent);
    GlossySpecular(real kspecular, real exponent, const RGBColor& color);



    // BRDF interface
public:
    RGBColor f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor sample_f(const ShadeRec &sr, const Vector &wi, const Vector &wo) const;
    RGBColor rho(const ShadeRec &sr, const Vector &wo) const;
};

#endif // GLOSSYSPECULAR_H
