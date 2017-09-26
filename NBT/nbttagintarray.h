#ifndef NBTTAGINTARRAY_H
#define NBTTAGINTARRAY_H
#include "nbttag.h"
#include <vector>

class NBTTagIntArray : public NBTTag
{
public:
    std::vector<int> _content;
public:
    NBTTagIntArray();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGINTARRAY_H
