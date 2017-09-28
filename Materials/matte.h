#ifndef MATTE_H
#define MATTE_H

#include "material.h"
#include "texture.h"

class Lambertian;

class Matte : public Material
{
protected:
    Lambertian* ambient_brdf;
    Lambertian* diffuse_brdf;

public:
    Matte();
    Matte(float ka_, float kd_, float r_, float g_, float b_);
    Matte(float ka_, float kd_, Texture* t);
    Matte(float ka_, float kd_, Texture* t, bool transparency_);

    void set_kambient(float k);
    void set_kdiffuse(float k);
    void set_color(float r, float g, float b);
    void set_color(Texture* t);

    // Material interface
public:
    RGBColor shade(ShadeRec &sr);

    // Material interface
public:
    RGBColor noshade(ShadeRec &sr);

    // Material interface
public:
    real transparency(const ShadeRec &sr) Q_DECL_OVERRIDE;
};

#endif // MATTE_H
