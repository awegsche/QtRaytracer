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
#include "mcgrid.h"
#include "phong.h"
#include "ambientoccluder.h"
#include "PureRandom.h"
#include "thinlens.h"

const real DIST = 20.0;
const int EDGE = 100;

void World::build() {
    vp.hres = 640;
    vp.vres = 480;
    vp.s = 1.0;
    vp.num_samples = 1;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new RayCast(this);

    Matte* mat = new Matte;
    mat->set_color(1,1,1);
    mat->set_kambient(1.5);
    mat->set_kdiffuse(.5);

    Grid* grid_ptr = new Grid();

//    for (int i = 0; i < EDGE; i++)
//        for (int j = 0; j < EDGE; j++)
//            for (int k = 0; k < EDGE; k++) {
//                Sphere* sph_ptr = new Sphere;

//                real x = (real)rand() / RAND_MAX * 10.0 - 5.0;
//                real y = (real)rand() / RAND_MAX * 10.0 - 5.0;
//                real z = (real)rand() / RAND_MAX * 10.0 - 5.0;

//                float r = (float)rand() / RAND_MAX;
//                float g = (float)rand() / RAND_MAX;
//                float b = (float)rand() / RAND_MAX;

//                sph_ptr->m = Point((real)i * DIST+ x, (real)j * DIST +y, (real)k * DIST +z);
//                sph_ptr->r = 1.0;
//                Matte* mat_ptr = new Matte;
//                mat_ptr->set_color(r,g,b);
//                mat_ptr->set_kambient(1.0);
//                mat_ptr->set_kdiffuse(1.0);
//                sph_ptr->set_material(mat_ptr);
//                //add_object(sph_ptr);
//                grid_ptr->add_object(sph_ptr);
//            }
/*
    MCGrid *mc_grid = new MCGrid();
    mc_grid->read_nbt(QString(""), this);
    //add_object(mc_grid);
    //grid_ptr->setup_cells();
    //grid_ptr->calculate_bounding_box();
    //add_object(grid_ptr);

    Matte* mat1 = new Matte(.8, .8, .8, .8, 1.0);
*/
//    Sphere *kugel = new Sphere(Point(400, 70, 60), 10);
//    Phong *phong = new Phong(.3, .5, .6, 2, 0,1,1);
//    kugel->set_material(phong);
//    add_object(kugel);

/*
    Plane* rear_ptr = new Plane();
    rear_ptr->set_casts_shadow(false);
    rear_ptr->normal = Normal(-1, 0, 0);
    rear_ptr->point = Point(10000, 0, 0);
    rear_ptr->set_material(mat1);
    add_object(rear_ptr);

    Plane* pl2 = new Plane();
    pl2->set_casts_shadow(false);

    pl2->normal = Normal(1, 0, 0);
    pl2->point = Point(-10000, 0, 0);
    pl2->set_material(mat1);
    add_object(pl2);

    Plane* pl3 = new Plane();
    pl3->set_casts_shadow(false);

    pl3->normal = Normal(0, 0, -1);
    pl3->point = Point(0, 0,10000);
    pl3->set_material(mat1);
    add_object(pl3);

    Plane* pl4 = new Plane();
    pl4->set_casts_shadow(false);

    pl4->normal = Normal(0, 0,1);
    pl4->point = Point(0, 0,-10000);
    pl4->set_material(mat1);
    add_object(pl4);

    Plane* floor = new Plane();
    floor->set_casts_shadow(false);

    floor->normal = Normal(0, 1, 0);
    floor->point = Point(0, -1, 0);
    floor->set_material(mat1);
    //add_object(floor);

*/
    PointLight* l = new PointLight(1.8,
                                  1.0, 1, 1,
                                  1000, 800, 2000);
    l->set_shadows(true);
    add_light(l);




}
