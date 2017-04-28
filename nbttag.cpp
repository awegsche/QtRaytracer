#include "nbttag.h"
#include "bigendianreader.h"
#include "nbttagbyte.h"
#include "nbttagbytearray.h"
#include "nbttagcompound.h"
#include "nbttagdouble.h"
#include "nbttagend.h"
#include "nbttagfloat.h"
#include "nbttagint.h"
#include "nbttagintarray.h"
#include "nbttaglist.h"
#include "nbttaglong.h"
#include "nbttagstring.h"

#include <vector>

NBTTag::NBTTag()
{

}

QString &NBTTag::Name()
{
    return _name;
}

void NBTTag::setName(const QString &name)
{
    _name = name;
}

NBTTag *fromFile(BigEndianReader &r)
{
    NBTTag::NBTTagID id = (NBTTag::NBTTagID)r.readByte();

    switch (id) {
    case NBTTag::TAG_Byte:{
        auto tag = new NBTTagByte();
        assignName(tag, r);
        tag->setValue(r.readByte());
        return tag;
    }
    case NBTTag::TAG_Byte_Array:{
        auto tag = new NBTTagByteArray();
        assignName(tag, r);
        int length = r.readInt32();
        tag->_content = r.readBytes(length);
        return tag;
    }
    case NBTTag::TAG_Compound:{
        NBTTagCompound *tag = new NBTTagCompound();
        assignName(tag, r);
        NBTTag *child;
        do{
            child = fromFile(r);
            tag->_children.push_back(child);
        } while (child->ID() != NBTTag::TAG_End);
        return tag;
    }
    case NBTTag::TAG_Double:{
        NBTTagDouble *tag = new NBTTagDouble;
        assignName(tag, r);
        tag->_value = r.readDouble();
        return tag;
    }
    case NBTTag::TAG_End:
        return new NBTTagEnd;
    case NBTTag::TAG_Float:{
        NBTTagFloat *tag = new NBTTagFloat;
        assignName(tag,r);
        tag->_value = r.readFloat();
        return tag;
    }
    case NBTTag::TAG_Int:{
        NBTTagInt *tag = new NBTTagInt;
        assignName(tag,r);
        tag->_value = r.readInt32();
        return tag;
    }
    case NBTTag::TAG_Int_Array:{
        auto tag = new NBTTagIntArray;
        assignName(tag, r);
        int size = r.readInt32();

        std::vector<int> array;
        array.reserve(size);

        for(int i = 0; i< size; i++)
            array.push_back(r.readInt32());
        tag->_content = array;
        return tag;
    }
    case NBTTag::TAG_List:{
        short length = r.readInt16_BigEndian();
        QString name = QString::fromUtf8(r.readBytes(length));
        NBTTag::NBTTagID id_ = (NBTTag::NBTTagID)r.readByte();
        int childcount = r.readInt32();
        switch(id_){
        case NBTTag::TAG_Compound:{
            NBTTagList<NBTTagCompound>* list = new NBTTagList<NBTTagCompound>;
            list->setName(name);
            for(int i = 0; i < childcount; i++){
                NBTTagCompound *tag = new NBTTagCompound();

                NBTTag *child;
                do{
                    child = fromFile(r);
                    tag->_children.push_back(child);
                } while (child->ID() != NBTTag::TAG_End);
                list->_children.push_back(tag);
            }
            return list;
        }
        case NBTTag::TAG_Double:{
            NBTTagList<NBTTagDouble>* list = new NBTTagList<NBTTagDouble>;
            list->setName(name);
            for(int i = 0; i < childcount; i++){
                NBTTagDouble *tag = new NBTTagDouble();
                tag->_value = r.readDouble();

                list->_children.push_back(tag);
            }
            return list;
        }
        case NBTTag::TAG_Float:{
            NBTTagList<NBTTagFloat>* list = new NBTTagList<NBTTagFloat>;
            list->setName(name);
            for(int i = 0; i < childcount; i++){
                NBTTagFloat *tag = new NBTTagFloat();
                tag->_value = r.readFloat();

                list->_children.push_back(tag);
            }
            return list;
        }

        case NBTTag::TAG_End:
            return new NBTTagList<NBTTagEnd>;
        }

        break;
    }
    case NBTTag::TAG_Long:{
        NBTTagLong *tag = new NBTTagLong;
        assignName(tag,r);
        tag->_value = r.readInt64();
        return tag;
    }
    case NBTTag::TAG_Short:{
        NBTTagInt *tag = new NBTTagInt;
        assignName(tag,r);
        tag->_value = r.readInt16_BigEndian();
        return tag;
    }
    case NBTTag::TAG_String:{
        NBTTagString* tag = new NBTTagString;
        assignName(tag,r);
        short stringlength = r.readInt16();
        tag->_value = QString::fromUtf8(r.readBytes(stringlength));
        return tag;
    }
    default:
        break;
    }
}

void assignName(NBTTag *tag, BigEndianReader &r)
{
    qint16 length = r.readInt16();
    if(length == 0)
        tag->setName("");
    else{
        QByteArray arr = r.readBytes(length);
        QString nme = QString::fromUtf8(arr);
        tag->setName(nme);
    }
}
