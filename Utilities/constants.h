#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <vectorclass.h>
#include "vector3d.h"

typedef Vec3d real3;
typedef double real;

typedef float CUDAreal;

typedef unsigned char byte;
const real kEpsilon = 1.0e-4;
const real kHugeValue = 1.0e+12;
const real Pi = 3.141592653589793238462643383279502;
const real invPi = 1.0 / Pi;
const real TWO_PI = 2 * Pi;
const real GRAD = Pi / 180.0;

// The edge length of one Minecraft block. If 1.0 the games coordinates will be equal to the Raytracer's world coordinates
const real BLOCKLENGTH = 1.0;




#endif // CONSTANTS_H
