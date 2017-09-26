#ifndef NBTTAG_H
#define NBTTAG_H
#include <QString>

class BigEndianReader;
class NBTTag
{
protected:
    QString _name;

public:
    enum NBTTagID{
        TAG_End = 0,
        TAG_Byte,
        TAG_Short,
        TAG_Int,
        TAG_Long,
        TAG_Float,
        TAG_Double,
        TAG_Byte_Array,
        TAG_String,
        TAG_List,
        TAG_Compound,
        TAG_Int_Array
    };

public:
    NBTTag();
    QString& Name();
    void setName(const QString& name);

    virtual NBTTagID ID() const = 0;
    //bool is_empty() override;
    virtual NBTTag *get_child(const QString &name);

    NBTTag *parent;

};

NBTTag* fromFile(BigEndianReader &r);

void assignName(NBTTag *tag, BigEndianReader& r);

#endif // NBTTAG_H
