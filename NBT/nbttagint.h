#ifndef NBTTAGINT_H
#define NBTTAGINT_H
#include "nbttag.h"

class NBTTagInt : public NBTTag
{
public:
    int _value;
public:
    NBTTagInt();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGINT_H
