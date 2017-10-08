#ifndef MATERIAL_H
#define MATERIAL_H

#include "constants.h"

class RGBColor;
class ShadeRec;

#ifdef WCUDA
class MaterialCUDA {
	virtual __device__ CUDAreal3 shade(ShadeRecCUDA& sr);
};
#endif // WCUDA


class Material
{
#ifdef WCUDA

protected:
	MaterialCUDA* device_ptr;

#endif // WCUDA

public:
    bool has_transparency;
public:
    Material();

    virtual RGBColor shade(ShadeRec& sr);
    virtual RGBColor area_light_shade(ShadeRec& sr);
    virtual RGBColor path_shade(ShadeRec& sr);
    virtual RGBColor noshade(ShadeRec& sr);
    virtual real transparency(const ShadeRec& sr);

	virtual MaterialCUDA* get_device_ptr();
};

#endif // MATERIAL_H
