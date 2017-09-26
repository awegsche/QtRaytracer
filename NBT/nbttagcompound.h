#ifndef NBTTAGCOMPOUND_H
#define NBTTAGCOMPOUND_H
#include "nbttag.h"
#include <vector>
#include <QString>

class NBTTagCompound : public NBTTag
{
public:
    std::vector<NBTTag*> _children;
public:
    NBTTagCompound();

//    void addChild(NBTTag* child);
//    NBTTag* getChild(const int index);
//    NBTTag* getChild(const QString& name);

    // NBTTag interface
public:
    NBTTagID ID() const;

    // NBTTag interface
public:
    NBTTag *get_child(const QString &name);
};

#endif // NBTTAGCOMPOUND_H
