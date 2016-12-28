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
};

#endif // TRACER_H
