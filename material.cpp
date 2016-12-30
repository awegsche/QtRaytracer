#include "material.h"
#include "rgbcolor.h"

Material::Material()
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
