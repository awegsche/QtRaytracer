#include "world.h"
#include "shaderec.h"
#include "point2d.h"
#include "camera.h"
#include "ambient.h"

//#include "simple_scene.cpp"

World::World(QObject *parent)
    : QThread(parent), camera_ptr(nullptr), ambient_ptr(new Ambient){

}

void World::add_object(GeometricObject *o)
{
    objects.push_back(o);
}

void World::add_light(Light *l)
{
    lights.push_back(l);
}

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

//void World::render_scene() const
//{

//}

ShadeRec World::hit_bare_bones_objects(const Ray &ray)
{
    ShadeRec sr(this);
    real t;
    real tmin = kHugeValue;
    int num_objects = objects.size();

    for (int j = 0; j < num_objects; j++)
        if (objects[j]->hit(ray, t, sr) && (t < tmin)) {
            sr.hit_an_object = true;
            tmin = t;
            //sr.color = objects[j]->get_color();
        }
    return sr;
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
            local_hit_point = sr.local_hit_point;
        }
    if (sr.hit_an_object) {
        sr.t = tmin;
        sr.normal = normal;
        sr.local_hit_point = local_hit_point;
        sr.material_ptr = mat_ptr;
    }

    return sr;
}

void World::dosplay_p(int r, int c, const RGBColor& pixel_color)
{
    RGBColor color = pixel_color.truncate();
    emit display_pixel(r, c, (int)(color.r * 255.0), (int)(color.g * 255.0),(int)(color.b * 255.0));
}

void World::run()
{
    render_camera();
}
