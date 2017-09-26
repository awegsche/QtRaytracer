#include "nbttagcompound.h"

NBTTagCompound::NBTTagCompound()
{

}

NBTTag::NBTTagID NBTTagCompound::ID() const
{
    return TAG_Compound;

}

NBTTag *NBTTagCompound::get_child(const QString &name)
{
    for (NBTTag *t : this->_children)
        if (t->Name() == name)
            return t;
    return nullptr;
}


