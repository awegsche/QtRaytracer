#ifndef NBTTAGBYTEARRAY_H
#define NBTTAGBYTEARRAY_H
#include "nbttag.h"
#include "QByteArray"

class NBTTagByteArray : public NBTTag
{
public:
    QByteArray _content;

public:
    NBTTagByteArray();


    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGBYTEARRAY_H
