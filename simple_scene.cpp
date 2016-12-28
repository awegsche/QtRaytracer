#include "world.h"
#include "ray.h"
#include "point.h"
#include "multipleobjects.h"

void World::build() {
    vp.hres = 200;
    vp.vres = 200;
    vp.s = 1.0;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new MultipleObjects(this);

    Sphere* sph_ptr = new Sphere;

    sph_ptr->m = Point(0, 0, -100);
    sph_ptr->r = 75;
    sph_ptr->color = RGBColor(1.0,0,0);
    add_object(sph_ptr);

}

