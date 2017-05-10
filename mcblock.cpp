#include "mcblock.h"
#include "shaderec.h"
#include "constants.h"
#include <math.h>

MCBlock::MCBlock()
    : air(false)
{

}

bool MCBlock::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    if (air) return false;

    sr.local_hit_point = ray.o + ray.d * tmin;

    switch(sr.hdir) {
    case ShadeRec::Top:
        sr.material_ptr = mat_top;
        sr.u = fmod(sr.local_hit_point.X, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z, BLOCKLENGTH);
        break;
    case ShadeRec::Bottom:
        sr.material_ptr = mat_side;
        sr.u = fmod(sr.local_hit_point.X, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z, BLOCKLENGTH);
        break;

    case ShadeRec::East:
        sr.material_ptr = mat_side;
        sr.u = fmod(sr.local_hit_point.X, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y, BLOCKLENGTH);
        break;

    case ShadeRec::West:
        sr.material_ptr = mat_side;
        sr.u = fmod(sr.local_hit_point.X, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y, BLOCKLENGTH);
        break;

    case ShadeRec::North:
        sr.material_ptr = mat_side;
        sr.u = fmod(sr.local_hit_point.Z, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y, BLOCKLENGTH);
        break;

    case ShadeRec::South:
        sr.material_ptr = mat_side;
        sr.u = fmod(sr.local_hit_point.Z, BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y, BLOCKLENGTH);
        break;
    default:
        sr.material_ptr = material_ptr;
        break;
    }

    if (sr.u < 0) sr.u = -sr.u;
    if (sr.v < 0) sr.v = -sr.v;
    sr.v = 1.0 - sr.v;


    sr.hitPoint = sr.local_hit_point;
    //sr.t = tmin;
    return true; // replace this by form-dependent (block, slab, plant) code

    return false;
}

bool MCBlock::shadow_hit(const Ray &ray, real &tmin) const
{
    if (air) return false;

    return true;

    return false;
}
