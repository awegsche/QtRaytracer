#include "world.h"
#include "ray.h"
#include "point.h"
#include "multipleobjects.h"
#include "pinhole.h"
#include "matte.h"
#include "pointlight.h"
#include "raycast.h"
#include "ambient.h"
#include "plane.h"

void World::build() {
    vp.hres = 400;
    vp.vres = 400;
    vp.s = 1.0;
    vp.num_samples = 16;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new RayCast(this);

    for (int i = 0; i < 15; i++)
        for (int j = 0; j < 15; j++) {
        Sphere* sph_ptr = new Sphere;

        real x = (real)rand() / RAND_MAX * 10.0;
        real y = (real)rand() / RAND_MAX * 10.0;

        float r = (float)rand() / RAND_MAX;
        float g = (float)rand() / RAND_MAX;
        float b = (float)rand() / RAND_MAX;

        sph_ptr->m = Point((real)i * 20+ x, (real)j * 20 +y, 0);
        sph_ptr->r = 5.0;
        Matte* mat_ptr = new Matte;
        mat_ptr->set_color(r,g,b);
        mat_ptr->set_kambient(1.0);
        mat_ptr->set_kdiffuse(1.0);
        sph_ptr->set_material(mat_ptr);
        add_object(sph_ptr);
    }
    Plane* pl_ptr = new Plane();

    pl_ptr->normal = Normal(0,1,0);
    pl_ptr->point = Point(0, 0, -20);
    Matte* mat = new Matte;
    mat->set_color(1,1,1);
    mat->set_kambient(.5);
    mat->set_kdiffuse(.5);
    pl_ptr->set_material(mat);
    //add_object(pl_ptr);


    PointLight* l = new PointLight(2.0,
                                  1.0, 1.0, 1.0,
                                  100, 100, 100);
    add_light(l);

    this->ambient_ptr = new Ambient(0.1, 1, 1, 1);

    Pinhole* ph_ptr = new Pinhole();
    ph_ptr->set_eye(0,50,1000);
    ph_ptr->set_lookat(150,150,0);
    ph_ptr->set_distance(1000);
    ph_ptr->set_zoom(2.0);
    ph_ptr->set_up(0,1,0);

    ph_ptr->compute_uvw();
    camera_ptr = ph_ptr;
}

