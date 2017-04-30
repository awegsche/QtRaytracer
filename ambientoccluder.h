#ifndef AMBIENTOCCLUDER_H
#define AMBIENTOCCLUDER_H

#include "light.h"
#include "rgbcolor.h"
#include "vector.h"
#include "sampler.h"

class AmbientOccluder : public Light
{
public:
    AmbientOccluder();
    AmbientOccluder(real ls_, real min_value, float r, float b, float g);

    // Light interface
public:
    Vector get_direction(ShadeRec &sr);
    RGBColor L(ShadeRec &sr);
    bool in_shadow(Ray &ray, ShadeRec &sr);

    void set_sampler(Sampler* sampler);

private:
    real ls;
    RGBColor color;
    Vector u, v, w;
    Sampler* sampler_ptr;
    real min_amount;
};

#endif // AMBIENTOCCLUDER_H
