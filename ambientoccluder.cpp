#include "ambientoccluder.h"
#include "point.h"
#include "shaderec.h"
#include "world.h"

AmbientOccluder::AmbientOccluder()
    : ls(1.0), color(1.0, 1.0, 1.0)
{

}

AmbientOccluder::AmbientOccluder(real ls_, real min_value, float r, float b, float g)
    : ls(ls_), color(r, g, b), min_amount(min_value)
{

}

Vector AmbientOccluder::get_direction(ShadeRec &sr)
{
//    Point sp = sampler_ptr->sample_hemisphere();

//    return sp.X * u + sp.Y * v + sp.Z * w;
    return Vector();
}

RGBColor AmbientOccluder::L(ShadeRec &sr)
{
    Vector u, v, w;
    w = sr.normal;
    v = w ^ Vector(-0.0073, 1.0, 0.0034);
    v.normalize();
    u = v ^ w;
    Point sp = sampler_ptr->sample_hemisphere();
    Vector get_d = sp.X * u + sp.Y * v + sp.Z * w;

    Ray shadow_ray(sr.local_hit_point + 0.001 * sr.normal, get_d);
    if (in_shadow(shadow_ray, sr))
        return min_amount * ls * color;
    else
        return ls * color;

}

bool AmbientOccluder::in_shadow(Ray &ray, ShadeRec &sr)
{
    real t;
    int num_objects = sr.w->objects.size();

    for (int j = 0; j < num_objects; j++)
        if (sr.w->objects[j]->shadow_hit(ray, t) && t > kEpsilon)
            return true;

    return false;
}

void AmbientOccluder::set_sampler(Sampler *sampler)
{
    sampler_ptr = sampler;
    sampler_ptr->generate_samples();
    sampler_ptr->map_samples_to_hemisphere(1.0);
}
