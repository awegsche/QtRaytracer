#include "imagetexture.h"

CUDAreal3 ImageTextureCUDA::get_color(const ShadeRecCUDA& sr) {
	int iu = (sr.u - kEpsilon) * width;
	int iv = (sr.v - kEpsilon) * height;

	return texels[iu + iv * width];
}