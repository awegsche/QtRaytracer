#ifndef THINLENSCUDA_H
#define THINLENSCUDA_H

#include "ray.h"

class MCWorldCUDA;


#ifdef WCUDA
extern "C" int render_thinlens_cuda(rayCU* rays, MCWorldCUDA* world,
    const int width, const int height, const int npixels, const CUDAreal vp_s,
    const int nsamples, const CUDAreal2 *disk_samples, const CUDAreal2 *square_samples,
    const CUDAreal aperture, const CUDAreal distance, const CUDAreal3 &eye, const CUDAreal3 &u, const CUDAreal3 &v, const CUDAreal3 &w);
#endif

class ThinLensCUDA
{
public:
    ThinLensCUDA();
};

#endif // THINLENSCUDA_H
