#ifndef TRACER_H
#define TRACER_H

#include "rgbcolor.h"
#include "ray.h"

class World;

class Tracer
{
public:
    World* world_ptr;
public:
    Tracer();
    Tracer(World* w_ptr);

    virtual RGBColor trace_ray(const Ray& ray) const;
    virtual RGBColor trace_ray(const Ray& ray, int depth) const;
    virtual RGBColor trace_ray(const Ray& ray, float& tmin, int depth) const;
};

#endif // TRACER_H
