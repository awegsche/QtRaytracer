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
        RGBColor L = world_ptr->background_color;
        if (noshade)
            L = sr.material_ptr->noshade(sr);
        else
            L = sr.material_ptr->shade(sr);
        if (sr.w->haze && sr.t > sr.w->haze_distance)
        {
            real damping = (sr.t - sr.w->haze_distance) * sr.w->haze_attenuation;
            damping = damping > 1.0 ? 1.0 : damping;
            return damping * sr.w->background_color + ((real)1.0 - damping) * L;
        }

        return L;
    }
    else
        return world_ptr->background_color;
}

#ifdef WCUDA



cudaError_t RayCast::trace_ray(rayCU * rays, CUDAreal3 * colours)
{
	
}
#endif // WCDUA