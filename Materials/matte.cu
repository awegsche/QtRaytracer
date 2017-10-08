#include "matte.h"
#include "shaderec.h"
#include "world.h"
#include "ray.h"

__device__ CUDAreal3 MatteCUDA::shade(ShadeRecCUDA& sr) {
	CUDAreal3 wo = -sr.ray.d;


	CUDAreal3 L = kd *  sr.w->ambient_ptr->L(sr);
	int numLights = sr.w->num_lights;

	for (int j = 0; j < numLights; j++) {
		CUDAreal3 wi = sr.w->lights[j]->get_direction(sr);
		CUDAreal ndotwi = sr.normal * wi;



		if (ndotwi > 0.0) {
			bool in_shadow = false;
			if (sr.w->lights[j]->casts_shadows())
			{
				rayCU shadowray = __make_CUDARay(sr.hit_point + kEpsilon * sr.normal, wi);
				in_shadow = sr.w->lights[j]->in_shadow(shadowray, sr);
			}

			if (!in_shadow)
				L += 
				//L += diffuse_brdf->f(sr, wo, wi) * sr.w->lights[j]->L(sr) * ndotwi;
		}
	}


	//if (has_transparency) {
	//	Ray second_ray(sr.local_hit_point + kEpsilon * sr.ray.d, sr.ray.d);

	//	real tr = diffuse_brdf->transparency(sr);
	//	if (tr < 1.0)
	//		L = tr * L + ((real)1.0 - tr) * sr.w->tracer_ptr->trace_ray(second_ray, sr.depth + 1);
	//}

	return L;
}