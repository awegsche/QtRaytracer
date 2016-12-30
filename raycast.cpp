#include "raycast.h"
#include "ray.h"
#include "shaderec.h"

RayCast::RayCast()
{

}

RGBColor RayCast::trace_ray(const Ray &ray, int depth) const
{
    ShadeRec sr(world_ptr->hit_objects(ray));

    if(sr.hit_an_object) {
        sr.ray = ray;
        return sr.material_ptr->shade(sr);
    }
    else
        return world_ptr->background_color;
}
