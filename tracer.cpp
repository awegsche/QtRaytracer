#include "tracer.h"

Tracer::Tracer()
    : world_ptr(nullptr){

}

Tracer::Tracer(World *w_ptr)
    : world_ptr(w_ptr){

}

RGBColor Tracer::trace_ray(const Ray &ray) const
{
    return RGBColor(0,0,0);
}
