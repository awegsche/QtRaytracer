#ifndef MATTE_H
#define MATTE_H

#include "material.h"

class Lambertian;

class Matte : public Material
{
private:
    Lambertian* ambient_brdf;
    Lambertian* diffuse_brdf;

public:
    Matte();
    Matte(float ka_, float kd_, float r_, float g_, float b_);

    void set_kambient(float k);
    void set_kdiffuse(float k);
    void set_color(float r, float g, float b);

    // Material interface
public:
    RGBColor shade(ShadeRec &sr);
};

#endif // MATTE_H
