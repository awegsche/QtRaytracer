#ifndef CHUNK_H
#define CHUNK_H
#include "bigendianreader.h"

class Chunk
{
private:

public:
    Chunk();

    static Chunk* emptyChunk();
};

#endif // CHUNK_H
