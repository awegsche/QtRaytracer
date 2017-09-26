#include "chunk.h"

Chunk::Chunk()
    :root(nullptr), x(0), y(0)
{

}

Chunk::Chunk(NBTTag *_root, int x_, int y_)
    :root(_root), x(x_), y(y_)
{

}

bool Chunk::is_empty()
{
    return root == nullptr;
}



Chunk *Chunk::emptyChunk()
{
    return new Chunk();
}
