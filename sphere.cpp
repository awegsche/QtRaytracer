#include "sphere.h"
#include "vector.h"
#include "ray.h"
#include "shaderec.h"

Sphere::Sphere()
{

}

Sphere::Sphere(Point center, real radius)
    : m(center), r(radius){

}

bool Sphere::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    real t;
    Vector temp = ray.o - m;
    real a = ray.d * ray.d;
    real b = 2.0 * temp * ray.d;
    real tempsq =  temp * temp;
    real c = tempsq - r * r;
    real disc = b * b - 4.0 * a * c;

    if (disc < 0.0)
        return false;
    else {
        real e = sqrt(disc);
        real denom = 2.0 * a;
        t = (-b - e) / denom;

        if(t > kEpsilon)
        {
            tmin = t;
            sr.normal = (temp + t * ray.d) / r;
            sr.local_hit_point = ray.o + t * ray.d;
            sr.material_ptr = material_ptr;
            return true;
        }

        t = (-b + e) / denom;

        if (t > kEpsilon) {
            tmin = t;
            sr.normal = (temp + t * ray.d) / r;
            sr.local_hit_point = ray.o + t * ray.d;
            sr.material_ptr = material_ptr;
            return true;
        }
    }
    return false;
}

BBox Sphere::get_bounding_box()
{
    return BBox(m.X - r - kEpsilon, m.Y - r - kEpsilon, m.Z - r - kEpsilon,
                m.X + r + kEpsilon, m.Y + r+ kEpsilon, m.Z + r + kEpsilon);
}

bool Sphere::shadow_hit(const Ray &ray, real &tmin) const
{
    real t;
    Vector temp = ray.o - m;
    real a = ray.d * ray.d;
    real b = 2.0 * temp * ray.d;
    real tempsq =  temp * temp;
    real c = tempsq - r * r;
    real disc = b * b - 4.0 * a * c;

    if (disc < 0.0)
        return false;
    else {
        real e = sqrt(disc);
        real denom = 2.0 * a;
        t = (-b - e) / denom;

        if(t > kEpsilon)
        {
            tmin = t;
            return true;
        }

        t = (-b + e) / denom;

        if (t > kEpsilon) {
            tmin = t;
            return true;
        }
    }
    return false;
}
