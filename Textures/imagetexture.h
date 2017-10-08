#ifndef IMAGETEXTURE_H
#define IMAGETEXTURE_H

#include "texture.h"
#include <QImage>
#include "rgbcolor.h"
#include <vector>
#include <QString>

#ifdef WCUDA
class ImageTextureCUDA : public TextureCUDA {
	CUDAreal3* texels;
	CUDAreal* transparency;

	int width, height;

	virtual __device__ CUDAreal3 get_color(const ShadeRecCUDA& sr) override;
};
#endif // WCUDA


class ImageTexture : public Texture
{
private:
    std::vector<RGBColor> texels;
    std::vector<real> transparency;
    int width;
    int height;
    QString m_filename;


public:
    ImageTexture();
    ImageTexture(const QString& filename, int sprite_width = 16, int sprite_height = 16);

    void colorize(const RGBColor& color_);


    // Texture interface
public:
    RGBColor get_color(const ShadeRec &sr);

    // Texture interface
public:
    real get_transparency(const ShadeRec &sr) Q_DECL_OVERRIDE;
};

#endif // IMAGETEXTURE_H
