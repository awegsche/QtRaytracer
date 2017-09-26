#ifndef REFLECTIVE_H
#define REFLECTIVE_H

#include "phong.h"
#include "perfectspecular.h"

class Reflective : public Phong
{
private:
    PerfectSpecular* reflective_brdf;
public:
    Reflective();

    void set_reflective_color(const real r, const real g, const real b);
    void set_kr(const real k);


    // Material interface
public:
    RGBColor shade(ShadeRec &sr) Q_DECL_OVERRIDE;
    RGBColor noshade(ShadeRec &sr) Q_DECL_OVERRIDE;

    // Material interface
public:
    real transparency(const ShadeRec &sr) Q_DECL_OVERRIDE;
};

#endif // REFLECTIVE_H
