#ifndef VIEWPLANE_H
#define VIEWPLANE_H

#include "sampler.h"

#ifdef WCUDA
#include "constants.h"
struct ViewPlaneCUDA {
	int hres, vres;
	CUDAreal s, gamma, inv_gamma;
	int num_samples;
};
#endif // WCUDA


class ViewPlane
{
public:
    int hres;
    int vres;
    float s;
    float gamma;
    float inv_gamma;
    int num_samples;
    Sampler* sampler_ptr;

public:
    ViewPlane();
    ViewPlane(const ViewPlane& vp);

    void set_gamma(float gamma_);
};

#endif // VIEWPLANE_H
