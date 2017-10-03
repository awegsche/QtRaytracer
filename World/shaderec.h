#ifndef SHADEREC_H
#define SHADEREC_H



#include "constants.h"
#include "point.h"
#include "normal.h"

#include "rgbcolor.h"

#include "material.h"


#include "ray.h"
#include "vector.h"

#ifdef WCUDA
struct ShadeRecCUDA {
	CUDAreal3 hit_point;
	void* material;
	CUDAreal3 normal;
	CUDAreal t;
};
#endif // WCUDA


class World;

class ShadeRec
{
public:
    enum HitDirection {
        North,
        South,
        East,
        West,
        Top,
        Bottom
    };

public:
    bool hit_an_object;
    Point  local_hit_point;     // texture coordinates
    Point hitPoint;             // world coordinates of hit point
    Material* material_ptr;     // material of nearest hit object
    Normal normal;
    Ray ray;                    // for speculiar highlights
    int depth;
    Vector dir;                 // for area lights
    real t;
    World *w;
    HitDirection hdir;
    real u;
    real v;
    real t_Before;


    ShadeRec(World *world);
    ShadeRec(const ShadeRec& sr);
};

#endif // SHADEREC_H
