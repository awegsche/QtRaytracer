#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <vectorclass.h>
#include "vector3d.h"

#ifdef WCUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#define __BOTH__ __device__ __host__

typedef float CUDAreal;
typedef float4 CUDAreal4;
typedef float3 CUDAreal3;
typedef float2 CUDAreal2;

static CUDAreal __constant__ kHugeValueCUDA = 1.0e+12;
static CUDAreal __constant__ PiCUDA = 3.141592653589793238462643383279502;
static CUDAreal  invPiCUDA = 1.0 / PiCUDA;
static CUDAreal  TWO_PI_CUDA = 2 * PiCUDA;
static CUDAreal  GRAD_CUDA = PiCUDA / 180.0;
static CUDAreal __constant__ BLOCKLENGTH_CUDA = 1.0;
#else
#define __BOTH__ __host__
#endif // WCUDA



typedef Vec3f real3;
typedef float real;


typedef unsigned char byte;

///<summary>
/// Everything smaller than kEpsilon will be considered zero.
/// Change kEpsilon to a smaller value (maybe 1.0e-7 for double).
///</summary>
const real kEpsilon = 1.0e-3; 

const real kHugeValue = 1.0e+12;
const real Pi = 3.141592653589793238462643383279502;
const real invPi = 1.0 / Pi;
const real TWO_PI = 2 * Pi;
const real GRAD = Pi / 180.0;

// The edge length of one Minecraft block. If 1.0 the games coordinates will be equal to the Raytracer's world coordinates
const real BLOCKLENGTH = 1.0;





#endif // CONSTANTS_H
