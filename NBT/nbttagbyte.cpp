#include "nbttagbyte.h"

NBTTagByte::NBTTagByte()
{

}

void NBTTagByte::setValue(const byte b)
{
    _value = b;
}

byte NBTTagByte::getValue() const
{
    return _value;
}

NBTTag::NBTTagID NBTTagByte::ID() const
{
    return NBTTagID::TAG_Byte;
}
