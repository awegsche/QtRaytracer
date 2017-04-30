#include "plane.h"
#include "shaderec.h"
#include "myexception.h"

Plane::Plane()
    :GeometricObject()
{

}

bool Plane::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    real t = (point - ray.o) * normal / (ray.d * normal);

    if (t > kEpsilon) {
        tmin = t;
        sr.normal = normal;
        sr.local_hit_point = ray.o + t * ray.d;
        sr.material_ptr = material_ptr;
        return true;
    }
    else
        return false;
}

BBox Plane::get_bounding_box()
{
    MyException e(QString("Plane does not have a bounding box since it is infinite."));
    e.raise();
}

bool Plane::shadow_hit(const Ray &ray, real &tmin) const
{
    if (!casts_shadow) return false;
    real t = (point - ray.o) * normal / (ray.d * normal);

    if (t > kEpsilon) {
        tmin = t;
        return true;
    }
    else
        return false;

}
