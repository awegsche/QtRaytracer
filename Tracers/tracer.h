#ifndef TRACER_H
#define TRACER_H

#include "rgbcolor.h"
#include "ray.h"
#include "material.h"
#include "noshadematte.h"

class World;

class Tracer
{
protected:
    NoShadeMatte* missing_mat;
    bool noshade;
public:
    World* world_ptr;
public:
    Tracer();
    Tracer(World* w_ptr);

    void set_shade(bool b);


    virtual RGBColor trace_ray(const Ray& ray) const;
    virtual RGBColor trace_ray(const Ray& ray, int depth) const;
    virtual RGBColor trace_ray(const Ray& ray, float& tmin, int depth) const;
};

#endif // TRACER_H
