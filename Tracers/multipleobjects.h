#ifndef MULTIPLEOBJECTS_H
#define MULTIPLEOBJECTS_H

#include "tracer.h"
#include "world.h"

class MultipleObjects : public Tracer
{
public:
    MultipleObjects();
    MultipleObjects(World* w);

    // Tracer interface
public:
    RGBColor trace_ray(const Ray &ray, int depth) const;
};

#endif // MULTIPLEOBJECTS_H
