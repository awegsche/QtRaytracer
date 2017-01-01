#include "geometricobject.h"
#include "rgbcolor.h"
#include "material.h"

GeometricObject::GeometricObject()
    : material_ptr(nullptr){

}

BBox GeometricObject::get_bounding_box()
{
    return BBox();
}

Material *GeometricObject::get_material()
{
    return material_ptr;
}


void GeometricObject::set_material(Material *mat)
{
    material_ptr = mat;
}
