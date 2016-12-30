#ifndef GEOMETRICOBJECT_H
#define GEOMETRICOBJECT_H

#include "constants.h"
#include "rgbcolor.h"

class Ray;
class ShadeRec;
class Material;


class GeometricObject
{
private:
    Material *material_ptr;
public:
    GeometricObject();

    virtual bool hit(const Ray& ray, real& tmin, ShadeRec &sr) const = 0;

    Material *get_material();

    void set_material(Material* mat);
};

#endif // GEOMETRICOBJECT_H
