#include "shaderec.h"

#include "point.h"
#include "vector.h"
#include "rgbcolor.h"


ShadeRec::ShadeRec(World *world)
    :   w(world), hit_an_object(false),
        material_ptr(nullptr), hitPoint(), local_hit_point(),
        normal(), ray(), depth(0), dir()
{

}

ShadeRec::ShadeRec(const ShadeRec &sr)
    :   w(sr.w), hit_an_object(sr.hit_an_object),
        material_ptr(sr.material_ptr), hitPoint(sr.hitPoint), local_hit_point(sr.local_hit_point),
        normal(sr.normal), ray(sr.ray), depth(sr.depth), dir(sr.dir)
{

}
