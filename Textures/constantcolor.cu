#include "constantcolor.h"

CUDAreal3 ConstantColorCUDA::get_color(const ShadeRecCUDA& sr) {
	return color;
}