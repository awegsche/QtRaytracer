#include "mcworldcuda.h"

ShadeRecCUDA MCWorldCUDA::hit_objects(const cuRay &Ray) const
{
    CUDAreal t = kEpsilonCUDA;
    ShadeRecCUDA sr;
    MCRegionGridCUDA.hit(ray, t, sr);

    return sr;
}

MCWorldCUDA::MCWorldCUDA()
{

}
