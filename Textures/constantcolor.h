#ifndef CONSTANTCOLOR_H
#define CONSTANTCOLOR_H

#include "texture.h"
#include "rgbcolor.h"

#ifdef WCUDA
class ConstantColorCUDA : public TextureCUDA {
	CUDAreal3 color;
	virtual __device__ CUDAreal3 get_color(const ShadeRecCUDA& sr) override;
};
#endif // WCUDA


class ConstantColor : public Texture
{
private:
    RGBColor color;
public:
    ConstantColor();
    ConstantColor(const RGBColor& color_);
    ConstantColor(const real r, const real g, const real b);

    // Texture interface
public:
    RGBColor get_color(const ShadeRec &sr);
};

#endif // CONSTANTCOLOR_H
