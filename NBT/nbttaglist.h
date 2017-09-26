#ifndef NBTTAGLIST_H
#define NBTTAGLIST_H
#include "nbttag.h"
#include <vector>

template<class T>
class NBTTagList : public NBTTag
{
public:
    std::vector<T*> _children;
public:
    NBTTagList()
    {

    }

    // NBTTag interface
public:
    NBTTagID ID() const
    {
        return TAG_List;
    }
    // NBTTagID Child_ID() const;
};

#endif // NBTTAGLIST_H
