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
#include "grid.h"
#include "constants.h"
#include "phong.h"

const real DIST = 20.0;
const int EDGE = 100;

void World::build() {
    vp.hres = 600;
    vp.vres = 400;
    vp.s = 1.0;
    vp.num_samples = 16;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new RayCast(this);
    Matte* mat = new Matte;
    mat->set_color(1,1,1);
    mat->set_kambient(1.5);
    mat->set_kdiffuse(.5);

    Grid* grid_ptr = new Grid(3.0);




    //add_object(grid_ptr);

    Sphere* s = new Sphere(Point(0,0,0), 20);
    Phong* p = new Phong(.2, .7, .05,5, 1,0,0);
    s->set_material(p);
    add_object(s);


    Plane* pl_ptr = new Plane();

    pl_ptr->normal = Normal(0,1,0);
    pl_ptr->point = Point(0, -10, 0);
    pl_ptr->set_material(mat);
    add_object(pl_ptr);


    PointLight* l = new PointLight(2.0,
                                  1.0, 1.0, 1.0,
                                  0, 300, -500);
    add_light(l);

    this->ambient_ptr = new Ambient(0.1, 1, 1, 1);

    Pinhole* ph_ptr = new Pinhole();
    ph_ptr->set_eye(0,300,500);
    ph_ptr->set_lookat(0,0,0);
    ph_ptr->set_distance(1000);
    ph_ptr->set_zoom(6);
    ph_ptr->set_up(0,1,0);

    ph_ptr->compute_uvw();
    camera_ptr = ph_ptr;
}

