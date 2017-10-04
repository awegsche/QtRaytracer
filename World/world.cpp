#include "world.h"
#include "shaderec.h"
#include "point2d.h"
#include "camera.h"
#include "ambient.h"
#include "nbttag.h"
#include "mcgrid.h"
#include "nbttagcompound.h"
#include "nbttaglist.h"
#include "nbttagint.h"
#include "nbttagbyte.h"
#include "nbttagbytearray.h"
#include "matte.h"
#include "constants.h"
#include "textureholder.h"
#include "matte.h"
#include "mcregiongrid.h"
#include "reflective.h"
#include "mcstandardblock.h"
#include "mcwaterblock.h"
#include "mcisoblock.h"


//#include "simple_scene.cpp"


World::World(QObject *parent)
    : QThread(parent), camera_ptr(nullptr), ambient_ptr(new Ambient), running(true), objects(),
      max_depth(4), haze(false), haze_distance(100.0), haze_attenuation(1.0e-2)

{
#if !defined NDEBUG && defined WCUDA
	mcgrid.nx = 0;
	mcgrid.ny = 0;
	mcgrid.nz = 0;
#endif
}

void World::add_object(GeometricObject *o)
{
    objects.push_back(o);
}

void World::add_light(Light *l)
{
    lights.push_back(l);
}

#ifdef WCUDA

WorldCUDA * World::get_device_world() const
{
	return dev_ptr;
}
WorldCUDA * World::setup_device_world()
{
	
	int size = this->objects.size();

	cudaMallocManaged(&dev_ptr, sizeof(int) + sizeof(GeometricObjectCUDA*));
	cudaMallocManaged(&dev_ptr->objects, size * sizeof(GeometricObjectCUDA**));

	dev_ptr->num_objects = size;

	for (int i = 0; i < size; i++)
		dev_ptr->objects[i] = objects[i]->get_device_ptr();

	return dev_ptr;
}
#endif // WCUDA


// obsolete
void World::render_scene_()
{
    RGBColor pixel_color;
    Ray ray;
    real zw = 1000.0;
    double x,y;
    int n = (int)sqrt((float)vp.num_samples);
    //Point2D pp;

    ray.d = Vector(0,0,-1.0);

    for(int row = 0; row < vp.vres; row++) {
        for(int column = 0; column < vp.hres; column++) {
            pixel_color = RGBColor(0,0,0);

            for (int p = 0; p < n; p++)
                for (int q = 0; q < n; q++) {


                    x = vp.s * (column - 0.5 * vp.hres + ((float) rand()) / (float) RAND_MAX);
                    y = vp.s * (row - 0.5 * vp.vres + ((float) rand()) / (float) RAND_MAX);
                    ray.o = Point(x,y,zw);
                    pixel_color += tracer_ptr->trace_ray(ray, 0);
                }
            pixel_color /= (float)vp.num_samples;
            emit display_pixel(row,column, (int)(pixel_color.r * 255.0), (int)(pixel_color.g * 255.0),(int)(pixel_color.b * 255.0) );
        }
    }
    emit done();
}

void World::render_camera()
{
    camera_ptr->render_scene(*this);
}


ShadeRec World::hit_objects(const Ray &ray)
{
    ShadeRec sr(this);
    real t = kHugeValue;
    Normal normal;
    Point local_hit_point;
    real tmin = kHugeValue;
    Material* mat_ptr;
    int num_objects = objects.size();

    for(int j = 0; j < num_objects; j++)
       if (objects[j]->hit(ray, t, sr) && t < tmin) {
            sr.hit_an_object = true;
            tmin = t;
            //sr.material_ptr = objects[j]->get_material(); // moved to hit function
            sr.hitPoint = ray.o + t * ray.d;
            mat_ptr = sr.material_ptr;
            normal = sr.normal;
            local_hit_point = sr.hitPoint;
        }
    if (sr.hit_an_object) {
        sr.t = tmin;
        sr.normal = normal;
        sr.local_hit_point = local_hit_point;
        sr.hitPoint = local_hit_point;
        sr.material_ptr = mat_ptr;
    }

    return sr;
}

void World::dosplay_p(int r, int c, const RGBColor& pixel_color)
{
    RGBColor color = pixel_color.truncate();

    emit display_pixel(r, c, (int)(color.r * 255.0), (int)(color.g * 255.0),(int)(color.b * 255.0));
}

ShadeRec * World::hit_objects_CUDA()
{
    //hit_test();
	return nullptr;
}



//Pixel World::display_p(Pixel& result, const Pixel &p)
//{
//    RGBColor color = p.color.truncate();
//    result = p;
//    emit p.w->display_pixel(p.point.Y, p.point.X, (int)(color.r * 255.0), (int)(color.g * 255.0),(int)(color.b * 255.0));
//    return p;
//}

void World::run()
{
    render_camera();
}
