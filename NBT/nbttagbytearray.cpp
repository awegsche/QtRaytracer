#include "nbttagbytearray.h"

NBTTagByteArray::NBTTagByteArray()
{

}

NBTTag::NBTTagID NBTTagByteArray::ID() const
{
    return NBTTagID::TAG_Byte_Array;
}
