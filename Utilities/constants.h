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



static __device__ bool inside_bb(const CUDAreal3 &p0, const CUDAreal3 &p1, const CUDAreal3 &point) {
	return
		point.x > p0.x && point.x < p1.x &&
		point.y > p0.y && point.y < p1.x &&
		point.z > p0.z && point.z < p1.z;
}

static __device__ CUDAreal clamp(CUDAreal value, CUDAreal a, CUDAreal b) {
	if (value < a) return a;
	if (value > b) return b;
	return value;
}

#else
// defining empty macros to be able to use __host__ and __device__ even in non-CUDA builds
#define __host__ 
#define __device__
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
