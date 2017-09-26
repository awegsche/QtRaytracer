#ifndef NBTTAGSTRING_H
#define NBTTAGSTRING_H
#include "nbttag.h"

class NBTTagString : public NBTTag
{
public:
    QString _value;
public:
    NBTTagString();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGSTRING_H
