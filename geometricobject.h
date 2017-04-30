#ifndef GEOMETRICOBJECT_H
#define GEOMETRICOBJECT_H

#include "constants.h"
#include "rgbcolor.h"
#include "bbox.h"

class Ray;
class ShadeRec;
class Material;


class GeometricObject
{
protected:
    Material *material_ptr;
    bool casts_shadow;
public:
    GeometricObject();

    virtual bool hit(const Ray& ray, real& tmin, ShadeRec &sr) const = 0;
    virtual bool shadow_hit(const Ray& ray, real& tmin) const = 0;
    virtual BBox get_bounding_box();

    void set_casts_shadow(bool b);

    Material *get_material();

    virtual void set_material(Material* mat);
};

#endif // GEOMETRICOBJECT_H
