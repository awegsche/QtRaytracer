#ifndef NBTTAGLONG_H
#define NBTTAGLONG_H
#include "nbttag.h"

class NBTTagLong : public NBTTag
{
public:
    long _value;
public:
    NBTTagLong();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGLONG_H
