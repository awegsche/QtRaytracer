#include "noshadematte.h"
#include "rgbcolor.h"
#include "ray.h"
#include "shaderec.h"
#include "lambertian.h"
#include "world.h"

NoShadeMatte::NoShadeMatte()
    : Matte()
{

}

NoShadeMatte::NoShadeMatte(float ka_, float kd_, float r_, float g_, float b_)
    : Matte(ka_, kd_, r_, g_, b_){

}

RGBColor NoShadeMatte::shade(ShadeRec &sr)
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
