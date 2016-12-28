#include "plane.h"
#include "shaderec.h"

Plane::Plane()
{

}

bool Plane::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    double t = (point - ray.o) * normal / (ray.d * normal);

    if (t > kEpsilon) {
        tmin = t;
        sr.normal = normal;
        sr.local_hit_point = ray.o + t * ray.d;

        return true;
    }
    else
        return false;
}
