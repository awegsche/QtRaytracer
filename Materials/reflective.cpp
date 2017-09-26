#include "reflective.h"
#include "shaderec.h"
#include "world.h"

Reflective::Reflective()
    : Phong(), reflective_brdf(new PerfectSpecular(1.0, 1.0, 1.0, 1.0))
{

}

void Reflective::set_reflective_color(const real r, const real g, const real b)
{
    reflective_brdf->set_color(r, g, b);
}

void Reflective::set_kr(const real k)
{
    reflective_brdf->set_kr(k);
}



RGBColor Reflective::shade(ShadeRec &sr)
{
    RGBColor L(Phong::shade(sr));

    Vector wo = -sr.ray.d;
    Vector wi;

    RGBColor fr = reflective_brdf->sample_f(sr, wi, wo);
    Ray reflected_ray(sr.local_hit_point + kEpsilon * sr.normal, wi);

    L += fr * sr.w->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1) * (sr.normal * wi);

    return L;
}

RGBColor Reflective::noshade(ShadeRec &sr)
{
    return Phong::noshade(sr);
}

real Reflective::transparency(const ShadeRec &sr)
{
    return diffuse_brdf->transparency(sr);
}
