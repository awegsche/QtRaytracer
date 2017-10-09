#ifndef MCWORLDCUDA_H
#define MCWORLDCUDA_H

#include "mcregiongrid.h"
#include "ray.h"

///<summary>
/// In order to use data oriented programming we need a new world class
/// that stores the objects accordingly.
///</summary>
class MCWorldCUDA
{
private:
    // Main Grid. Contains subgrids (linearly or as quadtrree, we will see).
    MCRegionGridCUDA main_grid;

    ThinLensCUDA* render_camera;

public:
    // hit the objects.
    __device__ ShadeRecCUDA hit_objects(const cuRay& Ray) const;

public:
    MCWorldCUDA();
};

#endif // MCWORLDCUDA_H
