#include "matte.h"
#include "lambertian.h"
#include "rgbcolor.h"
#include "shaderec.h"

Matte::Matte()
    : ambient_brdf(new Lambertian), diffuse_brdf(new Lambertian){

}

void Matte::set_kambient(float k)
{
    ambient_brdf->set_k(k);
}

void Matte::set_kdiffuse(float k)
{
    diffuse_brdf->set_k(k);
}

void Matte::set_color(float r, float g, float b)
{
    ambient_brdf->set_color(RGBColor(r, g, b));
    diffuse_brdf->set_color(RGBColor(r, g, b));
}

RGBColor Matte::shade(ShadeRec &sr)
{
    Vector wo = -sr.ray.d;
    RGBColor L = ambient_brdf->rho(sr, wo) * sr.w->ambient_ptr->L(sr);
    int numLights = sr.w->lights.size();

    for (int j = 0; j < numLights; j++) {
        Vector wi = sr.w->lights[j]->get_direction(sr);
        real ndotwi = sr.normal * wi;

        if(ndotwi > 0.0)
            L += diffuse_brdf->f(sr, wo, wi) * sr.w->lights[j]->L(sr) * ndotwi;
    }

    return L;
}
