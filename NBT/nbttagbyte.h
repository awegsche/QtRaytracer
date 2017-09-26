#ifndef NBTTAGBYTE_H
#define NBTTAGBYTE_H
#include "constants.h"
#include "nbttag.h"

class NBTTagByte : public NBTTag
{
public:
    byte _value;
public:
    NBTTagByte();

    void setValue(const byte b);
    byte getValue() const;

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGBYTE_H
