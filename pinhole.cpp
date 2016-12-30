#include "pinhole.h"
#include "rgbcolor.h"
#include "viewplane.h"
#include "ray.h"

Pinhole::Pinhole()
    :d(10.0), zoom(1.0){

}

Vector Pinhole::ray_direction(const Point2D &p) const
{
    Vector dir = p.X * u + p.Y * v - d * w;
    dir.normalize();
    return dir;
}

void Pinhole::render_scene(World &w)
{
       RGBColor L;
       ViewPlane vp(w.vp);
       Ray ray;
       int depth = 0;
       Point2D sp;      // Sample point in [0,1]x[0,1]
       Point2D pp;      // Sample point on a pixel

       vp.s /= zoom;
       ray.o = eye;

       for (int row = 0; row < vp.vres; row++) {
           for (int column = 0; column < vp.hres; column++) {
               L = RGBColor(0,0,0);

               for (int j = 0; j < vp.num_samples; j++) {
                   sp = vp.sampler_ptr->sample_unit_square();
                   pp.X = vp.s * (column - 0.5 * vp.hres + sp.X);
                   pp.Y = vp.s * (row - 0.5 * vp.vres + sp.Y);
                   ray.d = ray_direction(pp);
                   L += w.tracer_ptr->trace_ray(ray, depth);
               }
               L /= vp.num_samples;
               L *= exposure_time;
               w.dosplay_p(row, column, L);
           }

       }
}

void Pinhole::set_zoom(float z)
{
    zoom = z;
}

void Pinhole::set_distance(float distance)
{
    d = distance;
}
