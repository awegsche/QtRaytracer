#ifndef MATTE_H
#define MATTE_H

#include "material.h"
#include "texture.h"

class Lambertian;

#ifdef WCUDA
class MatteCUDA : public MaterialCUDA {
	// no BRDFs for now
public:
	CUDAreal ka, kd;
	TextureCUDA* texture;

	virtual __device__ CUDAreal3 shade(ShadeRecCUDA& sr) override;
};
#endif // WCUDA


class Matte : public Material
{
protected:
    Lambertian* ambient_brdf;
    Lambertian* diffuse_brdf;

public:
    Matte();
    Matte(float ka_, float kd_, float r_, float g_, float b_);
    Matte(float ka_, float kd_, Texture* t);
    Matte(float ka_, float kd_, Texture* t, bool transparency_);

    void set_kambient(float k);
    void set_kdiffuse(float k);
    void set_color(float r, float g, float b);
    void set_color(Texture* t);

    // Material interface
public:
    RGBColor shade(ShadeRec &sr);

    // Material interface
public:
    RGBColor noshade(ShadeRec &sr);

    // Material interface
public:
    real transparency(const ShadeRec &sr) Q_DECL_OVERRIDE;

	MatteCUDA* get_device_ptr() override;
};

#endif // MATTE_H
