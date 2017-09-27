#include "thinlens.h"

#include <qfuture.h>
#include <QtConcurrent/QtConcurrent>
#include "pixel.h"

#ifdef WCUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif // WCUDA



ThinLens::ThinLens()
    :_aperture(0.0)
{

}

ThinLens::ThinLens(const real eye_x, const real eye_y, const real eye_z,
                   const real lookat_x, const real lookat_y, const real lookat_z,
                   const real distance, const real zoom_, const real aperture)
    : Pinhole(eye_x, eye_y, eye_z, lookat_x, lookat_y, lookat_z, distance, zoom_),
      _aperture(aperture)
{

}

void ThinLens::set_sampler(Sampler *sampler)
{
    _sampler_ptr = sampler;
    _sampler_ptr->map_samples_to_unit_disk();
}

void ThinLens::render_scene_CUDA(World & w)
{
}

void ThinLens::render_line(ViewPlane vp, int row, World &w)
{
    Ray ray;

    Point2D sp;      // Sample point in [0,1]x[0,1]
    Point2D pp;      // Sample point on a pixel
    Point2D ap;     // Sample point on aperture;

    uint* rgb = new uint[vp.hres];

    for (int column = 0; column < vp.hres && w.running; column++) {
        int depth = 0;
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
        rgb[column] = L.truncate().to_uint();
    }
    emit w.display_line(row, rgb);
}

void ThinLens::render_scene(World &w)
{
    ViewPlane vp(w.vp);

    vp.s *= zoom;

#ifdef WCUDA // ======== GPU ===============================================================

	float* pixels;

	cudaMallocManaged(&pixels, vp.vres * vp.hres * 3 * sizeof(float));



#else // =============== CPU MultiCore =====================================================

	
    QThreadPool* pool = new QThreadPool();

    for (int row = 0; row < vp.vres && w.running; row++) {
        //render_line(vp, row, w);
        ThinLensLineRenderer *lr = new ThinLensLineRenderer(row, &w, vp, this);
        pool->start(lr);
    }

    pool->waitForDone(); // synchronize


#endif // WCUDA



    emit w.done();
}


ThinLensLineRenderer::ThinLensLineRenderer(const int line, World *w, const ViewPlane &vp, ThinLens *camera)
    :_line(line), _vp(vp), _w(w), _camera(camera)
{
}

ThinLensLineRenderer::~ThinLensLineRenderer()
{

}

void ThinLensLineRenderer::run()
{
    Ray ray;

    Point2D sp;      // Sample point in [0,1]x[0,1]
    Point2D pp;      // Sample point on a pixel
    Point2D ap;     // Sample point on aperture;

    uint* rgb = new uint[_vp.hres];


    for (int column = 0; column < _vp.hres && _w->running; column++) {
        int depth = 0;
        RGBColor L;
        Point2D pixel_point(column, _line);
        for (int j = 0; j < _vp.num_samples; j++) {
            sp = _vp.sampler_ptr->sample_unit_square();
            pp.X = _vp.s * (pixel_point.X - 0.5 * _vp.hres + sp.X);
            pp.Y = _vp.s * (pixel_point.Y - 0.5 * _vp.vres + sp.Y);

            ap = _camera->_sampler_ptr->sample_unit_disk();
            ray.o = _camera->eye + (_camera->_aperture * ap.X) * _camera->u + (_camera->_aperture * ap.Y) * _camera->v ;
            Vector dir = (pp.X - _camera->_aperture * ap.X) * _camera->u + (pp.Y - _camera->_aperture * ap.Y) * _camera->v - _camera->d * _camera->w;
            ray.d = dir.hat();
            //ray.d = Vector(30, 0.001, -5).hat();
            L += _w->tracer_ptr->trace_ray(ray, depth);
        }
        L /= _vp.num_samples;
        L *= _camera->exposure_time;
        rgb[column] = L.truncate().to_uint();
    }
    emit _w->display_line(_line, rgb);

}
