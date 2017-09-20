#include "phong.h"
#include "world.h"

Phong::Phong()
{

}

Phong::Phong(real kambient, real kdiffuse, real kspecular, real expspecular, float r, float g, float b)
    : ambient_brdf(new Lambertian(kambient, RGBColor(r,g,b))),
      diffuse_brdf(new Lambertian(kdiffuse, RGBColor(r,g,b))),
      specular_brdf(new GlossySpecular(kspecular, expspecular, RGBColor(r,g,b)))
{

}

void Phong::set_color(float r, float g, float b)
{

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
