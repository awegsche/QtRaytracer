#ifndef GEOMETRICOBJECT_H
#define GEOMETRICOBJECT_H

#include "constants.h"
#include "rgbcolor.h"
#include "bbox.h"
#include "shaderec.h"


class Ray;
class ShadeRec;
class Material;

#ifdef WCUDA
class rayCU;

class GeometricObjectCUDA
{
	virtual __device__ bool hit(const rayCU& ray, CUDAreal& tmin, ShadeRecCUDA &sr) const = 0;
	virtual __device__ bool shadow_hit(const rayCU& ray, CUDAreal& tmin) const = 0;
};
#endif // WCUDA



// Host Class
class GeometricObject
{
protected:
    Material *material_ptr;
    bool casts_shadow;
public:
    GeometricObject();
	~GeometricObject();

    virtual bool hit(const Ray& ray, real& tmin, ShadeRec &sr) const = 0;
    virtual bool shadow_hit(const Ray& ray, real& tmin) const = 0;
    virtual BBox get_bounding_box();

    void set_casts_shadow(bool b);

    Material *get_material();

    virtual void set_material(Material* mat);

#ifdef WCUDA
protected:
	GeometricObjectCUDA* device_ptr;
public:
	virtual GeometricObjectCUDA* get_device_ptr();

#endif // WCUDA


};

#endif // GEOMETRICOBJECT_H
