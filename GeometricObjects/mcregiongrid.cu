#include "mcregiongrid.h"



__device__ bool MCRegionGridCUDA::hit(const rayCU & ray, CUDAreal & tmin, ShadeRecCUDA & sr) const
{
	return false;
}

__device__ bool MCRegionGridCUDA::shadow_hit(const rayCU & ray, CUDAreal & tmin) const
{
	return false;
}
