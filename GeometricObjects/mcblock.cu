#include "mcblock.h"


__device__ bool MCBlockCUDA::hit(const rayCU & ray, CUDAreal & tmin, ShadeRecCUDA & sr) const
{
	return true;
}

__device__ bool MCBlockCUDA::shadow_hit(const rayCU & ray, CUDAreal & tmin) const
{
	return false;
}
