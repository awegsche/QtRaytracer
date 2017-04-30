#include "geometricobject.h"
#include "rgbcolor.h"
#include "material.h"

GeometricObject::GeometricObject()
    : material_ptr(nullptr), casts_shadow(true) {

}

BBox GeometricObject::get_bounding_box()
{
    return BBox();
}

void GeometricObject::set_casts_shadow(bool b)
{
    casts_shadow = b;
}

Material *GeometricObject::get_material()
{
    return material_ptr;
}


void GeometricObject::set_material(Material *mat)
{
    material_ptr = mat;
}
