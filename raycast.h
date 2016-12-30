#ifndef RAYCAST_H
#define RAYCAST_H

#include "tracer.h"
<<<<<<< HEAD
class World;
=======
>>>>>>> 5eababce84a924f7b3f281471cc8115b09966a0d

class RayCast : public Tracer
{
public:
    RayCast();
    RayCast(World *w);

    // Tracer interface
public:
    RGBColor trace_ray(const Ray &ray, int depth) const;
};

#endif // RAYCAST_H
