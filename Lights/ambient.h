#ifndef AMBIENT_H
#define AMBIENT_H

#include "light.h"
#include "rgbcolor.h"

#ifdef WCUDA

class AmbientCUDA : public LightCUDA {
public:
    CUDAreal ls;
    CUDAreal3 color;

    virtual __device__ CUDAreal3 L(ShadeRecCUDA& sr) override;
    virtual __device__ CUDAreal3 get_direction(ShadeRecCUDA& sr) override;
    virtual __device__ bool in_shadow(rayCU& ray, ShadeRecCUDA& sr) override;

};

#endif

class Ambient : public Light
{
private:
    float ls;
    RGBColor color;

public:
    Ambient();

    Ambient(float brightness, float r, float g, float b);

    // Light interface
public:
    Vector get_direction(ShadeRec &sr);
    RGBColor L(ShadeRec &sr);

    // Light interface
public:
    bool in_shadow(Ray& ray, ShadeRec& sr);


#ifdef WCUDA

public:
    AmbientCUDA *get_device_ptr() Q_DECL_OVERRIDE;

#endif
};

#endif // AMBIENT_H
