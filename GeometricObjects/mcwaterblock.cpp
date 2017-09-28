#include "mcwaterblock.h"
#include "shaderec.h"

MCWaterBlock::MCWaterBlock()
{

}

MCWaterBlock::MCWaterBlock(Material *material)
    : MCStandardBlock(material)
{

}

bool MCWaterBlock::shadow_hit(const Ray &ray, real &tmin) const
{
    if (tmin < kEpsilon) return false;
    return true;
}

bool MCWaterBlock::block_hit(const Ray &ray, const Point &p0, real &tmin, ShadeRec &sr) const
{
    if (tmin < kEpsilon) return false;




    sr.local_hit_point = ray.o + ray.d * tmin;
    sr.material_ptr = material_ptr;

    switch(sr.hdir) {
    case ShadeRec::Top:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        break;
    case ShadeRec::Bottom:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        break;

    case ShadeRec::East:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        break;

    case ShadeRec::West:
        sr.u = fmod(sr.local_hit_point.X(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        break;

    case ShadeRec::North:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        break;

    case ShadeRec::South:
        sr.u = fmod(sr.local_hit_point.Z(), BLOCKLENGTH);
        sr.v = fmod(sr.local_hit_point.Y(), BLOCKLENGTH);
        break;
    }

    if (sr.u < 0) sr.u = -sr.u;
    if (sr.v < 0) sr.v = -sr.v;
    sr.v = 1.0 - sr.v;

    sr.hitPoint = sr.local_hit_point;
    sr.t = tmin;

    return true; // replace this by form-dependent (block, slab, plant) code
}
