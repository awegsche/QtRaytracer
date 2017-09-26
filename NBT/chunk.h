#ifndef CHUNK_H
#define CHUNK_H
#include "bigendianreader.h"
#include "nbttag.h"

class Chunk
{
private:

public:
    Chunk();
    Chunk(NBTTag* _root, int x_, int y_);
    virtual bool is_empty();
    static Chunk* emptyChunk();
    NBTTag* root;
    int x, y;
};

#endif // CHUNK_H
