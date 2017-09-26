#ifndef NBTTAGDOUBLE_H
#define NBTTAGDOUBLE_H
#include "nbttag.h"


class NBTTagDouble : public NBTTag
{
public:
    double _value;
public:
    NBTTagDouble();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGDOUBLE_H
