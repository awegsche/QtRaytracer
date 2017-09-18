#include "textureholder.h"
#include <QString>
#include "rgbcolor.h"


TextureHolder::TextureHolder()
{
#ifndef WIN64 | WIN32
    QString texturepath = "/home/awegsche/Minecraft/minecraft/textures/blocks/";
#else
    QString texturepath = "G:\\Games\\Minecraft\\res\\minecraft\\textures\\blocks\\";
#endif


    textures.insert(1, new ImageTexture(texturepath + "stone.png"));
    textures.insert(2, new ImageTexture(texturepath + "grass_side.png"));
    textures.insert(12, new ImageTexture(texturepath + "sand.png"));
    textures.insert(60 + 1024, new ImageTexture(texturepath + "farmland_dry.png"));

    ImageTexture* t = new ImageTexture(texturepath + "grass_top.png");
    t->colorize(RGBColor(.0, 1.0, .0));
    textures.insert(2 + 1024, t);
    textures.insert(3, new ImageTexture(texturepath + "dirt.png"));
    textures.insert(17, new ImageTexture(texturepath + "log_oak.png"));
    textures.insert(17 + 1024, new ImageTexture(texturepath + "log_oak_top.png"));

    ImageTexture* t_leaves = new ImageTexture(texturepath + "leaves_oak.png");
    t_leaves->colorize(RGBColor(.0, 1.0, .0));
    textures.insert(18, t_leaves);

    textures.insert(37, new ImageTexture(texturepath + "flower_dandelion.png"));
    textures.insert(8, new ImageTexture(texturepath + "water_flow.png"));
    textures.insert(9, new ImageTexture(texturepath + "water_still.png"));


}

ImageTexture *TextureHolder::get_side(int blockid)
{
    if (textures.contains(blockid))
        return textures[blockid];
    return nullptr;
}

ImageTexture *TextureHolder::get_top(int blockid)
{
    if (textures.contains(blockid + 1024))
        return textures[blockid + 1024];
    else if (textures.contains(blockid))
        return textures[blockid];
    return nullptr;
}
