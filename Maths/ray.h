#ifndef RAY_H
#define RAY_H

#include "vector.h"
#include "point.h"

#ifdef WCUDA

struct rayCU {
	CUDAreal3 o, d;
};

///<summary>
/// initializes a new Ray on the device.
///</summary>
static __inline__ __host__ __device__ rayCU __make_CUDARay(const CUDAreal3 &o, const CUDAreal3 &d) {
	rayCU ray;
	ray.o = o;
	ray.d = d;
	return ray;
}


#endif



class Ray
{
public:
    Point o;
    Vector d;

public:
    Ray();
    Ray(const Point& origin, const Vector& dir);
    Ray(const Ray& ray);

    Ray& operator= (const Ray& ray);
};

#endif // RAY_H
