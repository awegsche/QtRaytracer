#include "pointlight.h"
#include "rgbcolor.h"
#include "vector.h"
#include "shaderec.h"
#include "world.h"

PointLight::PointLight()
    :ls(1.0), color(0.0), location(){

}

PointLight::PointLight(float brightness, float r, float g, float b, real x, real y, real z)
    :ls(brightness), color(r,g,b), location(x,y,z) {

}

void PointLight::set_brightness(float brightness)
{
    ls = brightness;
}

void PointLight::set_color(float r, float g, float b)
{
    color = RGBColor(r,g,b);
}

void PointLight::set_position(real x, real y, real z)
{
    location = Vector(x,y,z);
}

Vector PointLight::get_direction(ShadeRec &sr)
{

    return (location - sr.hitPoint).hat();
}

RGBColor PointLight::L(ShadeRec &sr)
{
    return ls * color;
}

bool PointLight::in_shadow(Ray& ray, ShadeRec& sr)
{
    real t = kHugeValue;
    int numObjects = sr.w->objects.size();
    real d = Vector(location - ray.o).length();


    for (int j = 0; j < numObjects; j++)
            if (sr.w->objects[j]->shadow_hit(ray, t) && t < d)
                    return (true);


    return (false);

}
