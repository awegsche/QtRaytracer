#include "tracer.h"
#include "matte.h"

Tracer::Tracer()
    : world_ptr(nullptr), missing_mat(new NoShadeMatte(.5, 1.0, 1.0, 1.0, 1.0)){

}

Tracer::Tracer(World *w_ptr)
    : world_ptr(w_ptr), missing_mat(new NoShadeMatte(.5, 1.0, 1.0, 1.0, 1.0)){

}

void Tracer::set_shade(bool b)
{
    noshade = b;
}

RGBColor Tracer::trace_ray(const Ray &ray) const
{
    return RGBColor();
}

RGBColor Tracer::trace_ray(const Ray &ray, int depth) const
{
    return RGBColor();
}

RGBColor Tracer::trace_ray(const Ray &ray, float &tmin, int depth) const
{
    return RGBColor();
}
