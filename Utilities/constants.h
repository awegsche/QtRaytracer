#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <vectorclass.h>
#include "vector3d.h"

#ifdef WCUDA
#include <cuda_runtime.h>
#endif // WCUDA

#include <vector_types.h>


typedef Vec3f real3;
typedef float real;

typedef float CUDAreal;
typedef float4 CUDAreal4;
typedef float3 CUDAreal3;
typedef float2 CUDAreal2;

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
