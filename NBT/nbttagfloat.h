#ifndef NBTTAGFLOAT_H
#define NBTTAGFLOAT_H
#include "nbttag.h"

class NBTTagFloat : public NBTTag
{
public:
    float _value;
public:
    NBTTagFloat();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGFLOAT_H
