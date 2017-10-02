#include "phong.h"
#include "world.h"
#include "constantcolor.h"

Phong::Phong()
    : ambient_brdf(new Lambertian((real)1.0, RGBColor((real).5))),
      diffuse_brdf(new Lambertian((real)1.0, RGBColor((real).5))),
      specular_brdf(new GlossySpecular((real)1.0, (real)1.0, RGBColor((real).5)))
{

}

Phong::Phong(real kambient, real kdiffuse, real kspecular, real expspecular, float r, float g, float b)
    : ambient_brdf(new Lambertian(kambient, RGBColor(r,g,b))),
      diffuse_brdf(new Lambertian(kdiffuse, RGBColor(r,g,b))),
      specular_brdf(new GlossySpecular(kspecular, expspecular, RGBColor(r,g,b)))
{

}

/// <summary>
/// Sets the ambient, diffuse and specular color to RGBColor(r,g,b)
/// </summary>
void Phong::set_color(const real r, const real g, const real b)
{
    ambient_brdf->set_color(RGBColor(r,g,b));
    diffuse_brdf->set_color(RGBColor(r,g,b));
    specular_brdf->set_color(new ConstantColor(r,g,b));
}

void Phong::set_ambient_color(const real r, const real g, const real b)
{
    ambient_brdf->set_color(RGBColor(r,g,b));
}

void Phong::set_diffuse_color(const real r, const real g, const real b)
{
    diffuse_brdf->set_color(RGBColor(r,g,b));
}

void Phong::set_specular_color(const real r, const real g, const real b)
{
    specular_brdf->set_color(new ConstantColor(r,g,b));
}

void Phong::set_ambient_color(Texture* t)
{
    ambient_brdf->set_color(t);
}

void Phong::set_diffuse_color(Texture* t)
{
    diffuse_brdf->set_color(t);
}

void Phong::set_specular_color(Texture* t)
{
    specular_brdf->set_color(t);
}

void Phong::set_ka(const real k)
{
    ambient_brdf->set_k(k);
}

void Phong::set_kd(const real k)
{
    diffuse_brdf->set_k(k);
}

void Phong::set_ks(const real k)
{
    specular_brdf->set_k(k);
}

void Phong::set_ks(Texture *t)
{
    specular_brdf->set_k(t);
}

void Phong::set_exp(const real e)
{
    specular_brdf->set_exp(e);
}

RGBColor Phong::shade(ShadeRec &sr)
{
    Vector wo = -sr.ray.d;
    RGBColor L = ambient_brdf->rho(sr, wo) * sr.w->ambient_ptr->L(sr);
    int num_lights = sr.w->lights.size();

    for (int j = 0; j < num_lights; j++) {
        Vector wi = sr.w->lights[j]->get_direction(sr);
        real ndotwi = sr.normal * wi;

        if (ndotwi > .0) {
            bool in_shadow = false;
            if(sr.w->lights[j]->casts_shadows())
            {
                Ray shadowray(sr.hitPoint + kEpsilon * sr.normal, wi);
                in_shadow = sr.w->lights[j]->in_shadow(shadowray, sr);
            }

            if(!in_shadow)
                L += (diffuse_brdf->f(sr, wo, wi) + specular_brdf->f(sr, wo, wi)) * sr.w->lights[j]->L(sr) * ndotwi;

        }
    }

    return L;
}

RGBColor Phong::noshade(ShadeRec &sr)
{
    Vector wo = -sr.ray.d;
    RGBColor L = ambient_brdf->rho(sr, wo) * sr.w->ambient_ptr->L(sr);
    int num_lights = sr.w->lights.size();

    for (int j = 0; j < num_lights; j++) {
        Vector wi = sr.w->lights[j]->get_direction(sr);
        real ndotwi = sr.normal * wi;

        if (ndotwi > .0) {

            L += (diffuse_brdf->f(sr, wo, wi) + specular_brdf->f(sr, wo, wi)) * sr.w->lights[j]->L(sr) * ndotwi;

        }
    }

    return L;
}
