#include "mcisoblock.h"
#include "shaderec.h"

MCIsoBlock::MCIsoBlock()
{

}

MCIsoBlock::MCIsoBlock(Material *top, Material *sides)
{
    material_ptr = sides;
    top_mat = top;
    bottom_mat = sides;
}

MCIsoBlock::MCIsoBlock(Material *top, Material *bottom, Material *sides)
{
    material_ptr = sides;
    top_mat = top;
    bottom_mat = bottom;
}

bool MCIsoBlock::hit(const Ray &ray, real &tmin, ShadeRec &sr) const
{
    sr.local_hit_point = ray.o + ray.d * tmin;


    switch(sr.hdir) {
    case ShadeRec::Top:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.material_ptr = top_mat;
        break;
    case ShadeRec::Bottom:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.material_ptr = bottom_mat;
        break;

    case ShadeRec::East:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::West:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::North:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::South:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;
    }

    if (sr.u < 0) sr.u = -sr.u;
    if (sr.v < 0) sr.v = -sr.v;
    sr.v = 1.0 - sr.v;


    sr.hitPoint = sr.local_hit_point;
    sr.t = tmin;

    return true; // replace this by form-dependent (block, slab, plant) code
}

bool MCIsoBlock::shadow_hit(const Ray &ray, real &tmin) const
{
    if ( tmin > kEpsilon) return true;
}

bool MCIsoBlock::block_hit(const Ray &ray, const Point &p0, real &tmin, ShadeRec &sr) const
{
    if (tmin < kEpsilon) return false;
    sr.local_hit_point = ray.o + ray.d * tmin;


    switch(sr.hdir) {
    case ShadeRec::Top:
        if (top_mat) {
            sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
            sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
            sr.material_ptr = top_mat;
        }
        else
            return false;
        break;
    case ShadeRec::Bottom:
        if (bottom_mat) {
            sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
            sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
            sr.material_ptr = bottom_mat;
        }
        else
            return false;
        break;

    case ShadeRec::East:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::West:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::North:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;

    case ShadeRec::South:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        sr.material_ptr = material_ptr;
        break;
    }

    if (sr.u < 0) sr.u = -sr.u;
    if (sr.v < 0) sr.v = -sr.v;
    sr.v = 1.0 - sr.v;


    sr.hitPoint = sr.local_hit_point;
    sr.t = tmin;

    return true; // replace this by form-dependent (block, slab, plant) code
}
