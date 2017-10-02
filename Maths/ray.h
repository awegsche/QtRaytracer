#ifndef RAY_H
#define RAY_H

#include "vector.h"
#include "point.h"

typedef struct {
	CUDAreal3 o, d;
} rayCU;



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
