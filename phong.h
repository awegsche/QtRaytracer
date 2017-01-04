#ifndef PHONG_H
#define PHONG_H

#include "material.h"
#include "lambertian.h"
#include "glossyspecular.h"


class Phong : public Material
{
protected:
    Lambertian* ambient_brdf;
    Lambertian* diffuse_brdf;
    GlossySpecular* specular_brdf;

public:
    Phong();
    Phong(real kambient, real kdiffuse, real kspecular, real expspecular, float r, float g, float b);

    void set_color(float r, float g, float b);


    // Material interface
public:
    RGBColor shade(ShadeRec &sr);
};

#endif // PHONG_H
