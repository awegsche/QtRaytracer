#include "compound.h"
#include <vector>
#include "constants.h"
#include "shaderec.h"
#include "point.h"
#include "matte.h"

Compound::Compound()
{

}

void Compound::add_object(GeometricObject *obj_ptr)
{
    objects.push_back(obj_ptr);
}

void Compound::calculate_bounding_box()
{
    Point p0(kHugeValue, kHugeValue, kHugeValue);
    Point p1(-kHugeValue, -kHugeValue, -kHugeValue);

    int num_objs = objects.size();

    for (int j = 0; j < num_objs; j++)
    {
        BBox bbox = objects[j]->get_bounding_box();

        if (bbox.x0 < p0.X()) p0.data.insert(0, bbox.x0);
        if (bbox.y0 < p0.Y()) p0.data.insert(1, bbox.y0);
        if (bbox.z0 < p0.Z()) p0.data.insert(2, bbox.z0);
        if (bbox.x1 > p1.X()) p1.data.insert(0, bbox.x1);
        if (bbox.y1 > p1.Y()) p1.data.insert(1, bbox.y1);
        if (bbox.z1 > p1.Z()) p1.data.insert(2,  bbox.z1);
    }

    boundingbox = BBox(p0 - Vector(kEpsilon), p1 + Vector(kEpsilon));
}

bool Compound::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    if (!boundingbox.hit(ray))
        return false;
    real t;
    Normal normal;
    Point local_hit_point;
    Point hitPoint;
    Material* mat_ptr;
    bool hit = false;

    int num_objects = objects.size();

    for(int j = 0; j < num_objects; j++) {
        if (objects[j]->hit(ray, t, sr) && (t < tmin)) {
            hit = true;
            tmin = t;
            mat_ptr = objects[j]->get_material();
            normal = sr.normal;
            hitPoint = sr.hitPoint;
        }
    }

    if(hit) {
        sr.t = tmin;
        sr.normal = normal;
        sr.local_hit_point = local_hit_point;
        sr.hitPoint = hitPoint;
        sr.material_ptr = mat_ptr;
//        Matte *matte = new Matte;
//        matte->set_color(1,0,1);
//        matte->set_kambient(5.0);
//        sr.material_ptr = matte;
    }

    return hit;
}

BBox Compound::get_bounding_box()
{
    return boundingbox;
}

bool Compound::shadow_hit(const Ray &ray, real &tmin) const
{
    if (!boundingbox.hit(ray))
        return false;
    real t;
    Normal normal;
    Point local_hit_point;
    Point hitPoint;
    bool hit = false;

    int num_objects = objects.size();

    for(int j = 0; j < num_objects; j++) {
        if (objects[j]->shadow_hit(ray, t) && (t < tmin)) {
            hit = true;
            tmin = t;
        }
    }
    return hit;

}

void Compound::set_material(Material *mat)
{
    int num_objs = objects.size();

    for (int j = 0; j < num_objs; j++)
    {
       objects[j]->set_material(mat);
    }

}

