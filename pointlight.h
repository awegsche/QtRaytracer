#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include "constants.h"
#include "light.h"
#include "rgbcolor.h"
#include "vector.h"
#include "point.h"


class PointLight : public Light
{
private:
    float ls;
    RGBColor color;
    Point location;


public:
    PointLight();
    PointLight(float brightness, float r, float g, float b, real x, real y, real z);

    void set_brightness(float brightness);
    void set_color(float r, float g, float b);
    void set_position(real x, real y, real z);

    // Light interface
public:
    Vector get_direction(ShadeRec &sr);
    RGBColor L(ShadeRec &sr);
};

#endif // POINTLIGHT_H
