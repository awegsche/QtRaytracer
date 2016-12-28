#include "ray.h"

Ray::Ray()
{

}

Ray::Ray(const Point &origin, const Vector &dir)
    : o(origin), d(dir) {

}

Ray::Ray(const Ray &ray)
    : o(ray.o), v(ray.d) {

}

Ray &Ray::operator=(const Ray &ray)
{
    this->d = ray.d;
    this->o = ray.o;

    return *this;
}
