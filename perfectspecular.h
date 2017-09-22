#ifndef PERFECTSPECULAR_H
#define PERFECTSPECULAR_H

#include "brdf.h"
#include "texture.h"

class PerfectSpecular : public BRDF
{
private:
    Texture* kr;
    Texture* cr;
public:
    PerfectSpecular();
    PerfectSpecular(const real cr_red, const real cr_green, const real cr_blue, const real _kr);

    void set_color(const real r, const real g, const real b);
    void set_kr(const real k);
    void set_kr(Texture* t);

    // BRDF interface
public:
    RGBColor sample_f(const ShadeRec &sr, Vector &wi, const Vector &wo) const Q_DECL_OVERRIDE;
};

#endif // PERFECTSPECULAR_H
