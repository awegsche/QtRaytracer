#include "imagetexture.h"
#include <QImage>

ImageTexture::ImageTexture()
{

}

ImageTexture::ImageTexture(const QString &filename, int sprite_width, int sprite_height)
{
    QImage i;

    if (i.load(filename, "png")) {

        m_filename = filename;

        height = i.height();
        width = i.width();

        texels.resize(width * height);
        transparency.resize(texels.size());

        for(int x = 0; x < width; x++)
            for(int y = 0; y < height; y++)
            {
                QRgb col = i.pixel(x,y);
                texels[x + y * width] = RGBColor(col);
                transparency[x + y * width] = (byte)((col & 0xFF000000) >> 24);
            }
//        width = sprite_width;
//        height = sprite_height;
    }
    else
    {
        width = 1;
        height = 1;
        texels.push_back(RGBColor(1,0,1));
    }

}

void ImageTexture::colorize(const RGBColor &color_)
{
    int count = texels.size();

    for (int i = 0; i < count; i++)
        texels[i] *= color_;
}

RGBColor ImageTexture::get_color(const ShadeRec &sr)
{
    int iu = sr.u * width;
    int iv = sr.v * height;

    return texels[iu + iv * width];
}
