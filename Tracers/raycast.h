#ifndef RAYCAST_H
#define RAYCAST_H

#include "tracer.h"
class World;


class RayCast : public Tracer
{

public:
    RayCast();
    RayCast(World *w);




    // Tracer interface
public:
    RGBColor trace_ray(const Ray &ray, const int depth) const;
};

#endif // RAYCAST_H
