#include "mcworld.h"
#include "nbttag.h"
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
#include <QVariant>


MCWorld::MCWorld():
    _width(3200), _chunks()
{

}

void MCWorld::addChunk(const int xkey, const int ykey, NBTTag *chunk)
{
    int key = ykey*_width + xkey;
    _chunks.push_back(new Chunk(chunk, xkey, ykey));
}

//Chunk *MCWorld::chunkAt(const int xkey, const int ykey)
//{
//     int key = ykey*_width + xkey;
//    if(_chunks.contains(key))
//        return _chunks[key];
//    return Chunk::emptyChunk();
//}

QModelIndex MCWorld::index(int row, int column, const QModelIndex &parent) const
{
    if(!parent.isValid())
    {
        // wir befinden uns im root.

        return createIndex(row, column, _chunks[row]->root);
    }

    // wenn nicht, dann ist parent ein NBTTag;

    NBTTag* tag = (NBTTag*)parent.internalPointer();


    switch(tag->ID()){ // manche NBTTags haben children:
    case NBTTag::TAG_Byte_Array:{
        NBTTagByteArray* temp = (NBTTagByteArray*)tag;

        return createIndex(row, column, new QVariant(temp->_content[row]));

    }
    case NBTTag::TAG_Compound:{
        return createIndex(row, column, ((NBTTagCompound*)tag)->_children[row]);
    }
    case NBTTag::TAG_Int_Array:{
        return createIndex(row, column, new QVariant(((NBTTagIntArray*)tag)->_content[row]));
    }
      case NBTTag::TAG_List:{
        return createIndex(row, column, ((NBTTagList<NBTTag>*)tag)->_children[row]);
    }
    }
    return QModelIndex();
}

QModelIndex MCWorld::parent(const QModelIndex &child) const
{
    if (child.isValid() && ((NBTTag*)child.internalPointer())->parent != nullptr) {
        return createIndex(0, 0, ((NBTTag*)child.internalPointer())->parent);
    }
    return QModelIndex();
}

int MCWorld::rowCount(const QModelIndex &parent) const
{
    if(!parent.isValid())
    {
        // wir befinden uns im root.

        return _chunks.size();
    }

    // wenn nicht, dann ist parent ein NBTTag;

    NBTTag* tag = (NBTTag*)parent.internalPointer();


    switch(tag->ID()){ // manche NBTTags haben children:
    case NBTTag::TAG_Byte_Array:{
        NBTTagByteArray* temp = (NBTTagByteArray*)tag;

        return temp->_content.size();
    }
    case NBTTag::TAG_Compound:{
        return ((NBTTagCompound*)tag)->_children.size();
    }
    case NBTTag::TAG_Int_Array:{
        return ((NBTTagIntArray*)tag)->_content.size();
    }
      case NBTTag::TAG_List:{
        return ((NBTTagList<NBTTag>*)tag)->_children.size();
    }
    }

    return 0;
}

int MCWorld::columnCount(const QModelIndex &parent) const
{
    return 2;
}

QVariant MCWorld::data(const QModelIndex &index, int role) const
{
    //index sollte auf jeden Fall NBTTag sein oder invalid

    if(!index.isValid()) return QVariant();


    NBTTag* tag = (NBTTag*)index.internalPointer();
    if (role == Qt::DisplayRole) {
        switch(tag->ID()){
        case NBTTag::TAG_Byte:{
            NBTTagByte *temp = (NBTTagByte*)tag;

            if(index.column() == 0)
                return temp->Name();
            if(index.column() == 1)
                return QVariant(temp->_value);

            }

        case NBTTag::TAG_End:
            return "End";
        default:{
            if (index.column() == 0)
                return tag->Name();
            if (index.column() == 1)
                return tag->ID();
        }
        }
    }

    return QVariant();
}
