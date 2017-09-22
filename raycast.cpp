#include "raycast.h"
#include "ray.h"
#include "shaderec.h"
#include "matte.h"
#include "world.h"

RayCast::RayCast()
{
    noshade = false;
}

RayCast::RayCast(World *w)
    :Tracer(w) {
    noshade = false;
}



RGBColor RayCast::trace_ray(const Ray &ray, const int depth) const
{
    if (depth > this->world_ptr->max_depth) return RGBColor();
    ShadeRec sr(world_ptr->hit_objects(ray));

    if(sr.hit_an_object) {
        sr.depth = depth;
        sr.ray = ray;
        if (sr.material_ptr == nullptr) sr.material_ptr = missing_mat;
        if (noshade)
            return sr.material_ptr->noshade(sr);
        return sr.material_ptr->shade(sr);
    }
    else
        return world_ptr->background_color;
}
