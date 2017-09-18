#include "matte.h"
#include "lambertian.h"
#include "rgbcolor.h"
#include "shaderec.h"
#include "world.h"

Matte::Matte()
    : ambient_brdf(new Lambertian), diffuse_brdf(new Lambertian){

}

Matte::Matte(float ka_, float kd_, float r_, float g_, float b_)
    : ambient_brdf(new Lambertian), diffuse_brdf(new Lambertian){
    ambient_brdf->set_k(ka_);
    diffuse_brdf->set_k(kd_);
    ambient_brdf->set_color(RGBColor(r_,g_,b_));
    diffuse_brdf->set_color(RGBColor(r_,g_,b_));
}

Matte::Matte(float ka_, float kd_, Texture *t)
    : ambient_brdf(new Lambertian), diffuse_brdf(new Lambertian){
    ambient_brdf->set_k(ka_);
    diffuse_brdf->set_k(kd_);
    ambient_brdf->set_color(t);
    diffuse_brdf->set_color(t);
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

void Matte::set_color(Texture *t)
{
    ambient_brdf->set_color(t);
    diffuse_brdf->set_color(t);
}

RGBColor Matte::shade(ShadeRec &sr)
{
    Vector wo = -sr.ray.d;
    RGBColor L = ambient_brdf->rho(sr, wo) * sr.w->ambient_ptr->L(sr);
    int numLights = sr.w->lights.size();

    for (int j = 0; j < numLights; j++) {
        Vector wi = sr.w->lights[j]->get_direction(sr);
        real ndotwi = sr.normal * wi;



        if(ndotwi > 0.0) {
            bool in_shadow = false;
            if(sr.w->lights[j]->casts_shadows())
            {
                Ray shadowray(sr.local_hit_point + wi*0.001, wi);
                in_shadow = sr.w->lights[j]->in_shadow(shadowray, sr);
            }

            if(!in_shadow)
                L += diffuse_brdf->f(sr, wo, wi) * sr.w->lights[j]->L(sr) * ndotwi;
        }
    }

    return L;
}

RGBColor Matte::noshade(ShadeRec &sr)
{
    Vector wo = -sr.ray.d;
    RGBColor L = ambient_brdf->rho(sr, wo) * sr.w->ambient_ptr->L(sr);
    int numLights = sr.w->lights.size();

    for (int j = 0; j < numLights; j++) {
        Vector wi = sr.w->lights[j]->get_direction(sr);
        real ndotwi = sr.normal * wi;



        if(ndotwi > 0.0) {

            L += diffuse_brdf->f(sr, wo, wi) * sr.w->lights[j]->L(sr) * ndotwi;
        }
    }

    return L;
}
