#ifdef WCUDA



#include <cuda_runtime.h>
#include "ray.cuh"
#include "ray.h"
#include <curand_kernel.h>
#include "CUDAhelpers.h"
#include "shaderec.h"
#include "world.h"

const int CUDABLOCK = 8;

static __global__ void thinlens_kernel(
	rayCU* rays, WorldCUDA* world,
	const int width, const int height, const int npixels, const CUDAreal vp_s, 
	const int nsamples, const CUDAreal2 *disk_samples, const CUDAreal2 *square_samples,
	const CUDAreal aperture, const CUDAreal distance, const CUDAreal3 eye, const CUDAreal3 u, const CUDAreal3 v, const CUDAreal3 w) {

	int column = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int index = row * width + column;
		
	for (int i = 0; i < nsamples; i++) {
		float2 pp;
		float2 sp = square_samples[i + threadIdx.x + threadIdx.y]; // not really random
		pp.x = vp_s * (column - 0.5 * width + sp.x);
		pp.y = vp_s * (row - 0.5 * height + sp.y);

		float2 ap = disk_samples[i + threadIdx.x + threadIdx.y];

		int index_ray = index + i * npixels;
		rays[index_ray].o = eye + (aperture * ap.x) * u + (aperture * ap.y) * v;
		CUDAreal3 dir = (pp.x - aperture * ap.x) * u + (pp.y - aperture * ap.y)  * v - distance * w;
		rays[index_ray].d = normalize(dir);

		ShadeRecCUDA sr;
		CUDAreal t = kHugeValueCUDA;

		/*if (mcgrid_hit_kernel(rays[index_ray], mcgrid, &sr, &t)) {
			rays[index_ray].d = sr.normal;
		}
		else*/
			rays[index_ray].d = __make_CUDAreal3(-3.14, 0,0);
		




	}
}

static __global__ void thinlens_trace_kernel(
	rayCU* rays, 
	const int width, const int height, const int npixels, const CUDAreal vp_s,
	const int nsamples, const CUDAreal2 *disk_samples, const CUDAreal2 *square_samples,
	const CUDAreal aperture, const CUDAreal distance, const CUDAreal3 eye, const CUDAreal3 u, const CUDAreal3 v, const CUDAreal3 w)
{
}

// Setup the array of primary rays to render
extern "C" int render_thinlens_cuda(rayCU* rays, MCGridCUDA* mcgrid,
	const int width, const int height, const int npixels, const CUDAreal vp_s,
	const int nsamples, const CUDAreal2 *disk_samples, const CUDAreal2 *square_samples,
	const CUDAreal aperture, const CUDAreal distance, const CUDAreal3 &eye, const CUDAreal3 &u, const CUDAreal3 &v, const CUDAreal3 &w)
{
	dim3 blockSize(CUDABLOCK, CUDABLOCK);
	dim3 numBlocks(width / CUDABLOCK, height / CUDABLOCK);

	cudaDeviceProp p;

	cudaGetDeviceProperties(&p, 0);
	
	thinlens_kernel<<<numBlocks, blockSize>>>(
		rays, mcgrid,
		width, height, npixels,  vp_s,
		nsamples, square_samples, disk_samples,
		aperture, distance, eye, u, v, w);

	return cudaDeviceSynchronize();




	/*RGBColor RayCast::trace_ray(const Ray &ray, const int depth) const
	{
		if (depth > this->world_ptr->max_depth) return RGBColor();
		ShadeRec sr(world_ptr->hit_objects(ray));

		if (sr.hit_an_object) {
			sr.depth = depth;
			sr.ray = ray;
			if (sr.material_ptr == nullptr) sr.material_ptr = missing_mat;
			RGBColor L = world_ptr->background_color;
			if (noshade)
				L = sr.material_ptr->noshade(sr);
			else
				L = sr.material_ptr->shade(sr);
			if (sr.w->haze && sr.t > sr.w->haze_distance)
			{
				real damping = (sr.t - sr.w->haze_distance) * sr.w->haze_attenuation;
				damping = damping > 1.0 ? 1.0 : damping;
				return damping * sr.w->background_color + ((real)1.0 - damping) * L;
			}

			return L;
		}
		else
			return world_ptr->background_color;
	}*/

}
#endif // WCUDA