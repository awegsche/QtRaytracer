#ifdef WCUDA

#include "world.h"
#include "shaderec.h"
#include "cuda_runtime.h"


__device__ ShadeRecCUDA WorldCUDA::hit_objects(const rayCU& ray) {
	ShadeRecCUDA sr;
	CUDAreal t = kHugeValueCUDA;
	bool hit = false;
	for (int j = 0; j < num_objects; j++) {
		if (objects[j]->hit(ray, t, sr)) {
			sr.hit_point = ray.o + t * ray.d;
		} 
	}

	return sr;
}

#endif