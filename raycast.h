#ifndef RAYCAST_H
#define RAYCAST_H

#include "tracer.h"

class RayCast : public Tracer
{
public:
    RayCast();

    // Tracer interface
public:
    RGBColor trace_ray(const Ray &ray, int depth) const;
};

#endif // RAYCAST_H
