#ifndef TEXTUREHOLDER_H
#define TEXTUREHOLDER_H

#include "imagetexture.h"
#include <QMap>

class TextureHolder
{
private:
    QMap<int, ImageTexture*> textures;
public:
    TextureHolder();

    ImageTexture* get_side(int blockid);
    ImageTexture* get_top(int blockid);
};

#endif // TEXTUREHOLDER_H
