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

const real DIST = 20.0;
const int EDGE = 100;

void World::build() {
    vp.hres = 400;
    vp.vres = 400;
    vp.s = 1.0;
    vp.num_samples = 1;

    background_color = RGBColor(0,0,0);

    tracer_ptr = new RayCast(this);
    Matte* mat = new Matte;
    mat->set_color(1,1,1);
    mat->set_kambient(1.5);
    mat->set_kdiffuse(.5);

    Grid* grid_ptr = new Grid(3.0);

    grid_ptr->read_ply_file("E:\\Andreas\\3D Modelle\\Tomb Raider\\laracroft\\RenderMesh\\Section 1076out.ply", 0);
    //grid_ptr->compute_mesh_normals();
    grid_ptr->set_material(mat);
    grid_ptr->setup_cells();
    //grid_ptr->calculate_bounding_box();
    add_object(grid_ptr);


    Plane* pl_ptr = new Plane();

    pl_ptr->normal = Normal(0,1,0);
    pl_ptr->point = Point(0, -10, 0);
    pl_ptr->set_material(mat);
    add_object(pl_ptr);


    PointLight* l = new PointLight(2.0,
                                  1.0, 1.0, 1.0,
                                  100, 100, 100);
    add_light(l);

    this->ambient_ptr = new Ambient(0.1, 1, 1, 1);

    Pinhole* ph_ptr = new Pinhole();
    ph_ptr->set_eye(0,300,500);
    ph_ptr->set_lookat(0,100,0);
    ph_ptr->set_distance(1000);
    ph_ptr->set_zoom(3);
    ph_ptr->set_up(0,1,0);

    ph_ptr->compute_uvw();
    camera_ptr = ph_ptr;
}

