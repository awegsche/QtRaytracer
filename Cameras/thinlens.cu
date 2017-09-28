#include "thinlens.h"
#include <cuda_runtime.h>

__device__ void render_thinlens_cuda()
{
	//Ray ray;

	//Point2D sp;      // Sample point in [0,1]x[0,1]
	//Point2D pp;      // Sample point on a pixel
	//Point2D ap;     // Sample point on aperture;

	//uint* rgb = new uint[_vp.hres];


	//for (int column = 0; column < _vp.hres && _w->running; column++) {
	//	int depth = 0;
	//	RGBColor L;
	//	Point2D pixel_point(column, _line);
	//	for (int j = 0; j < _vp.num_samples; j++) {
			//sp = _vp.sampler_ptr->sample_unit_square();
	//		pp.X = _vp.s * (pixel_point.X - 0.5 * _vp.hres + sp.X);
	//		pp.Y = _vp.s * (pixel_point.Y - 0.5 * _vp.vres + sp.Y);

	//		ap = _camera->_sampler_ptr->sample_unit_disk();
	//		ray.o = _camera->eye + (_camera->_aperture * ap.X) * _camera->u + (_camera->_aperture * ap.Y) * _camera->v;
	//		Vector dir = (pp.X - _camera->_aperture * ap.X) * _camera->u + (pp.Y - _camera->_aperture * ap.Y) * _camera->v - _camera->d * _camera->w;
	//		ray.d = dir.hat();
	//		//ray.d = Vector(30, 0.001, -5).hat();
	//		L += _w->tracer_ptr->trace_ray(ray, depth);
	//	}
	//	L /= _vp.num_samples;
	//	L *= _camera->exposure_time;
	//	rgb[column] = L.truncate().to_uint();
	//}
	//emit _w->display_line(_line, rgb);
}