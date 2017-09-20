#include "thinlens.h"

ThinLens::ThinLens()
    :_aperture(0.0)
{

}

void ThinLens::set_sampler(Sampler *sampler)
{
    _sampler_ptr = sampler;
    _sampler_ptr->map_samples_to_unit_disk();
}

void ThinLens::render_scene(World &w)
{
    ViewPlane vp(w.vp);
    Ray ray;
    int depth = 0;
    Point2D sp;      // Sample point in [0,1]x[0,1]
    Point2D pp;      // Sample point on a pixel
    Point2D ap;     // Sample point on aperture;

    vp.s /= zoom;


    for (int row = 0; row < vp.vres && w.running; row++) {
        for (int column = 0; column < vp.hres && w.running; column++) {
            RGBColor L;
            Point2D pixel_point(column, row);
            for (int j = 0; j < vp.num_samples; j++) {
                sp = vp.sampler_ptr->sample_unit_square();
                pp.X = vp.s * (pixel_point.X - 0.5 * vp.hres + sp.X);
                pp.Y = vp.s * (pixel_point.Y - 0.5 * vp.vres + sp.Y);

                ap = _sampler_ptr->sample_unit_disk();
                ray.o = eye + (_aperture * ap.X) * u + (_aperture * ap.Y) * v ;
                Vector dir = (pp.X - _aperture * ap.X) * u + (pp.Y - _aperture * ap.Y) * v - d * this->w;
                ray.d = dir.hat();
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
