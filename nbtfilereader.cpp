#include "nbtfilereader.h"
#include "nbttag.h"
#include "bigendianreader.h"
#include "chunk.h"
#include "world.h"
#include <QDebug>
#include <QFile>
#include "QDataStream"
NBTFileReader::NBTFileReader(const QString &filename)
    :_filename(filename)
{
    //normally: get x and z from filename
    //but now I load only the file -1.0, so

    _X = -1;
    _Z = 0;

//    FILE* file = std::fopen(filename.toStdString().c_str(), "rb");
//    char* buffer = new char[file->_bufsiz];
//    fgets(buffer, file->_bufsiz, file);

//    std::fclose(file);




}

void NBTFileReader::Load(MCWorld *world)
{
    QFile file(_filename);
    file.open(QIODevice::ReadOnly);
    //byte* buffer = new byte[1024];
    QByteArray buffer =  file.read(4096);

    BigEndianReader Lreader(buffer);
    //QDataStream stream(&file);

    // OK, nun kommt das NBT file format:
    // die ersten 1024 * 4 byte sind Chunk-offset informationen

    for(int i = 0; i< 1024; i++){




       int offset = Lreader.readInt24();
       Lreader.readByte();
       if (offset == 0) continue;
       //stream.skipRawData(1);

        qDebug() << "\noffset["<<i<<"] = "<<offset<<"";

        Chunk* c = new Chunk();


        file.seek(offset*4096);


        byte* tmp = new byte[4];
        file.read((char*)tmp, 4);

        qint32 length = readInt32(tmp, 0);

        qDebug() << "length = " << length;

        file.seek(offset*4096+5);

       auto size = BigEndianReader::toByteArray(length);

        QByteArray _chunkdata = file.read(length+1);

        auto chunkdata = size.append(_chunkdata);
        QByteArray unc_chunkdata = qUncompress(chunkdata);
        BigEndianReader R(unc_chunkdata);

        NBTTag *root = fromFile(R);

        world->addChunk(i%32, i/32, root);
       // w->add_chunk(root, _X*32 + i%32, _Z*32 + i/32);


    }

    file.close();

}

qint64 NBTFileReader::readInt64(byte *src, int position)
{
    qint64 i = 0;
    i |= src[position] << 56;
    i |= src[position + 1] << 48;
    i |= src[position + 2] << 40;
    i |= src[position + 3] << 32;
    i |= src[position + 4] << 24;
    i |= src[position + 5] << 16;
    i |= src[position + 6] << 8;
    i |= src[position + 7];

    return i;
}

qint64 NBTFileReader::readInt64_BigEndian(byte *src, int position)
{
    qint64 i = 0;
    i |= src[position + 7] << 56;
    i |= src[position + 6] << 48;
    i |= src[position + 5] << 40;
    i |= src[position + 4] << 32;
    i |= src[position + 3] << 24;
    i |= src[position + 2] << 16;
    i |= src[position + 1] << 8;
    i |= src[position];

    return i;
}

qint16 NBTFileReader::readInt16(byte *src, int position)
{
    qint16 i = 0;

    i |= src[position] << 8;
    i |= src[position + 1];

    return i;
}
qint32 NBTFileReader::readInt24(byte *src, int position)
{
    qint32 i = 0;
    i |= (int)src[position] << 16;
    i |= (int)src[position + 1] << 8;
    i |= (int)src[position + 2];

    return i;
}
qint32 NBTFileReader::readInt32(byte *src, int position)
{
    qint32 i = 0;
    i |= (uint)src[position] << 24;
    i |= (uint)src[position + 1] << 16;
    i |= (uint)src[position + 2] << 8;
    i |= (uint)src[position + 3];

    return i;
}

