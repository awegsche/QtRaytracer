#include "material.h"
#include "rgbcolor.h"

Material::Material()
    :has_transparency(false)
{

}

RGBColor Material::shade(ShadeRec &sr)
{
    return RGBColor();
}

RGBColor Material::area_light_shade(ShadeRec &sr)
{
    return RGBColor();
}

RGBColor Material::path_shade(ShadeRec &sr)
{
    return RGBColor();
}

RGBColor Material::noshade(ShadeRec &sr)
{
    return RGBColor();
}

real Material::transparency(const ShadeRec &sr)
{
    return 1.0;
}
