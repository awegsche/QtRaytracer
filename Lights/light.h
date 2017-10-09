#ifndef LIGHT_H
#define LIGHT_H

#include "ray.h"

class Vector;
class RGBColor;
class ShadeRec;

#ifdef WCUDA
class ShadeRecCUDA;

// Device Light class.
class LightCUDA {
public:
	virtual __device__ CUDAreal3 L(ShadeRecCUDA& sr) = 0;
	virtual __device__ CUDAreal3 get_direction(ShadeRecCUDA& sr) = 0;
	virtual __device__ bool in_shadow(rayCU& ray, ShadeRecCUDA& sr) = 0;

	bool __device__ casts_shadows();

protected:
	bool shadows;
};

bool __inline__ __device__ LightCUDA::casts_shadows() {
	return shadows;
}
#endif // WCUDA


// Represents a light in the scene. Responsible for direct illumination and shadows.
class Light
{

public:
    Light();

    // Returns the direction in which the light can be seen starting from the current hit point.
    // For light sources like spherical light this will be sampled.
    virtual Vector get_direction(ShadeRec& sr) = 0;

    // Returns the color of the light source as seen from the hitpoint.
    virtual RGBColor L(ShadeRec& sr) = 0;

    bool casts_shadows();
    void set_shadows(bool b);

    // Returns true if the hit point lies in the shadow of this light source.
    virtual bool in_shadow(Ray& ray, ShadeRec& sr) = 0;

protected:
    bool shadows;

#ifdef WCUDA
protected:
	LightCUDA* device_ptr;

public:
	LightCUDA* get_device_ptr();
#endif // WCUDA

};

#endif // LIGHT_H
