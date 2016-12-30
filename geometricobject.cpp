#include "geometricobject.h"
#include "rgbcolor.h"

GeometricObject::GeometricObject()
    : material_ptr(nullptr){

}

Material *GeometricObject::get_material()
{
    return material_ptr;
}

