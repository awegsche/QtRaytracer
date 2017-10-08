#include "material.h"
#include "shaderec.h"
#include "CUDAhelpers.h"

CUDAreal3 MaterialCUDA::shade(ShadeRecCUDA& sr) {
	return __make_CUDAreal3(0, 0, 0);
}