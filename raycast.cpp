#include "raycast.h"
#include "ray.h"
#include "shaderec.h"
#include "matte.h"

RayCast::RayCast()
{

}

RayCast::RayCast(World *w)
    :Tracer(w){

}

RGBColor RayCast::trace_ray(const Ray &ray, int depth) const
{
    ShadeRec sr(world_ptr->hit_objects(ray));

    if(sr.hit_an_object) {
        sr.ray = ray;
        if (sr.material_ptr == nullptr) sr.material_ptr = missing_mat;
        return sr.material_ptr->shade(sr);
    }
    else
        return world_ptr->background_color;
}
