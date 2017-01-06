#include "concurrentstruct.h"

ConcurrentStruct::ConcurrentStruct()
{

}

ConcurrentStruct::ConcurrentStruct(Ray &r_, Point2D &pp_, ViewPlane &vp_, Point2D &sp_, int depth_, RGBColor &l_, World *w_, Point2D &pix_point_)
    : ray(r_), pp(pp_), sp(sp_), vp(vp_), depth(depth_), L(l_), w(w_), pixel_point(pix_point_){

}
