#ifndef NBTTAGEND_H
#define NBTTAGEND_H
#include "nbttag.h"

class NBTTagEnd : public NBTTag
{
public:
    NBTTagEnd();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGEND_H
