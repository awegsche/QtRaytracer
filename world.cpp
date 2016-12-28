#include "world.h"
#include "shaderec.h"

//#include "simple_scene.cpp"

World::World(QObject *parent)
    : QThread(parent){

}

void World::add_object(GeometricObject *o)
{
    objects.push_back(o);
}

void World::render_scene_()
{
    RGBColor pixel_color;
    Ray ray;
    real zw = 1000.0;
    double x,y;

    ray.d = Vector(0,0,-1.0);

    for(int row = 0; row < vp.vres; row++) {
        for(int column = 0; column < vp.hres; column++) {
            x = vp.s * (column - 0.5 * (vp.hres - 1.0));
            y = vp.s * (row - 0.5 * (vp.vres - 1.0));
            ray.o = Point(x,y,zw);
            pixel_color = tracer_ptr->trace_ray(ray);
            emit display_pixel(row,column, (int)(pixel_color.r * 255.0), (int)(pixel_color.g * 255.0),(int)(pixel_color.b * 255.0) );
        }
    }
    emit done();
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
            sr.color = objects[j]->get_color();
        }
    return sr;
}

void World::run()
{
    render_scene_();
}
