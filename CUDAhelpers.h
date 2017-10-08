#pragma once

#ifdef WCUDA



#include "constants.h"

#include <vector_functions.h>

///<summary>
/// initializes a new CUDAreal3. Independent of the exact type of CUDAreal.
///</summary>
static __inline__ __host__ __device__ CUDAreal3 __make_CUDAreal3(CUDAreal x, CUDAreal y, CUDAreal z) {
	CUDAreal3 temp;
	temp.x = x;
	temp.y = y;
	temp.z = z;
	return temp;
}



static __inline__ __host__ __device__ CUDAreal2 __make_CUDAreal2(CUDAreal x, CUDAreal y) {
	CUDAreal2 temp;
	temp.x = x;
	temp.y = y;
	return temp;
}


// Multiplication of a vector v with v real number a
static __inline__ __host__ __device__ CUDAreal3 operator*(const CUDAreal3& v, const CUDAreal a) {
	return __make_CUDAreal3(v.x * a, v.y * a, v.z * a);
}

// Multiplication of a vector v with v real number a
static __inline__ __host__ __device__ CUDAreal3 operator*(const CUDAreal a, const CUDAreal3& v) {
	return __make_CUDAreal3(v.x * a, v.y * a, v.z * a);
}

// Addition of two vectors a and b
static __inline__ __host__ __device__ CUDAreal3 operator+(const CUDAreal3& a, const CUDAreal3& b) {
	return __make_CUDAreal3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Difference of two vectors a and b
static __inline__ __host__ __device__ CUDAreal3 operator-(const CUDAreal3& a, const CUDAreal3& b) {
	return __make_CUDAreal3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Scalar product
static __inline__ __host__ __device__ CUDAreal operator*(const CUDAreal3& a, const CUDAreal3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Negative of a vector a
static __inline__ __host__ __device__ CUDAreal3 operator-(const CUDAreal3& a) {
	return __make_CUDAreal3(-a.x , -a.y ,-a.z );
}

// Returns the normalized vector of v
static __inline__ __host__ __device__ CUDAreal3 normalize(const CUDAreal3& v) {
	CUDAreal over_l = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return __make_CUDAreal3(v.x * over_l, v.y * over_l, v.z * over_l);
}


#endif // WCUDA
