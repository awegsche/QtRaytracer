#include "pinhole.h"
#include "rgbcolor.h"
#include "viewplane.h"
#include "ray.h"
#include <QFuture>
#include "pixel.h"
#include <QList>


Pinhole::Pinhole()
    :d(10.0), zoom(1.0){

}

Pinhole::Pinhole(const Camera &cam)
    : Pinhole()
{
    // if general camera, let d and zoom at default and copy u
    this->u = cam.u;
    this->v = cam.v;
    this->w = cam.w;
    this->exposure_time = cam.exposure_time;

    try
    {

        this->d = ((Pinhole&)cam).d;
        this->zoom = ((Pinhole&)cam).zoom;
    }
    catch(std::exception &e)
    {
        this->d = 10.0;
        this->zoom = 1.0;

    }
}

Pinhole::Pinhole(const real eye_x, const real eye_y, const real eye_z,
                 const real lookat_x, const real lookat_y, const real lookat_z,
                 const real distance, const real zoom_)
    : Camera(eye_x, eye_y, eye_z, lookat_x, lookat_y, lookat_z), d(distance), zoom(zoom_)
{

}


Vector Pinhole::ray_direction(const Point2D &p) const
{
    Vector dir = p.X * u + p.Y * v - d * w;
    dir.normalize();
    return dir;
}

void Pinhole::render_scene(World &w)
{

       ViewPlane vp(w.vp);
       Ray ray;
       int depth = 0;
       Point2D sp;      // Sample point in [0,1]x[0,1]
       Point2D pp;      // Sample point on a pixel

       vp.s /= zoom;
       ray.o = eye;


       for (int row = 0; row < vp.vres && w.running; row++) {
           for (int column = 0; column < vp.hres && w.running; column++) {
               RGBColor L;
               Point2D pixel_point(column, row);
               for (int j = 0; j < vp.num_samples; j++) {
                   sp = vp.sampler_ptr->sample_unit_square();
                   pp.X = vp.s * (pixel_point.X - 0.5 * vp.hres + sp.X);
                   pp.Y = vp.s * (pixel_point.Y - 0.5 * vp.vres + sp.Y);
                   ray.d = ray_direction(pp);
                   //ray.d = Vector(30, 0.001, -5).hat();
                   L += w.tracer_ptr->trace_ray(ray, depth);
               }
               L /= vp.num_samples;
               L *= exposure_time;
               w.dosplay_p(pixel_point.Y, pixel_point.X, L);
           }
       }

       /*/


       QList<ConcurrentStruct> pixel_points;
       pixel_points.reserve(vp.vres*vp.hres);

       for (int row = 0; row < vp.vres; row++) {
           for (int column = 0; column < vp.hres; column++) {
               Point2D pixel_point(column, row);


               ConcurrentStruct cs = ConcurrentStruct(ray, pp, vp, sp, depth, L, &w, pixel_point);
               pixel_points.push_back(cs);


//               L = render_pixel(cs);
//               w.dosplay_p(pixel_point.Y, pixel_point.X, L);
           }

       }

       QFuture<Pixel> pixels = QtConcurrent::mappedReduced(pixel_points, &Pinhole::render_pixel, &World::display_p);


       // */

       emit w.done();
}

void Pinhole::set_zoom(double z)
{
    zoom = z;
}

double Pinhole::get_zoom() const
{
    return zoom;
}

void Pinhole::rescale_zoom(double a)
{
    zoom *= a;
}

void Pinhole::set_distance(double distance)
{
    d = distance;
}
