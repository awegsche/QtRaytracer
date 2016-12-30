#include "world.h"
#include "ray.h"
#include "point.h"
#include "multipleobjects.h"
#include "pinhole.h"

void World::build() {
    vp.hres = 200;
    vp.vres = 200;
    vp.s = 1.0;
    vp.num_samples = 16;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new MultipleObjects(this);

    Sphere* sph_ptr = new Sphere;

    sph_ptr->m = Point(0, 0, 0);
    sph_ptr->r = 75;
    sph_ptr->color = RGBColor(1.0,0,0);
    add_object(sph_ptr);

    Pinhole* ph_ptr = new Pinhole();
    ph_ptr->set_eye(0,0,100);
    ph_ptr->set_lookat(0,0,0);
    ph_ptr->set_distance(100);
    ph_ptr->set_zoom(1.0);

    ph_ptr->compute_uvw();
    camera_ptr = ph_ptr;
}

