#ifndef COMPOUND_H
#define COMPOUND_H

#include "geometricobject.h"
#include "bbox.h"

class Compound : public GeometricObject
{
protected:
    BBox boundingbox;
    std::vector<GeometricObject*> objects;

public:
    Compound();

    void add_object(GeometricObject *obj_ptr);
    void calculate_bounding_box();

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &tmin, ShadeRec &sr) const;
    BBox get_bounding_box();

    // GeometricObject interface
public:
    bool shadow_hit(const Ray &ray, real &tmin) const;
    void set_material(Material *mat);
};

#endif // COMPOUND_H
