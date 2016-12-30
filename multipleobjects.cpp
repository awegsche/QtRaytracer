#include "multipleobjects.h"
#include "shaderec.h"

MultipleObjects::MultipleObjects()
{

}

MultipleObjects::MultipleObjects(World *w)
    : Tracer(w){

}

RGBColor MultipleObjects::trace_ray(const Ray &ray, int depth) const
{
    ShadeRec sr(world_ptr->hit_bare_bones_objects(ray));

    if(sr.hit_an_object)
        return sr.color;
    else
        return world_ptr->background_color;
}
