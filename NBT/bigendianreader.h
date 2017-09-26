#ifndef BIGENDIANREADER_H
#define BIGENDIANREADER_H
#include <QByteArray>
#include "constants.h"



class BigEndianReader
{
private:
    union{
        short value;
        byte array[2];
    }shortu;

    union{
        int value;
        byte array[4];
    }intu;

    union{
        long value;
        byte array[8];
    }longu;

    union{
        float value;
        byte array[4];
    } floatu;

    union{
        double value;
        byte array[8];
    }doubleu;
protected:
    int _position;
    QByteArray _stream;
public:
    BigEndianReader(QByteArray &stream);

    short readInt16();
    int readInt24();
    int readInt32();
    long readInt64();
    short readInt16_BigEndian();
    int readInt24_BigEndian();
    int readInt32_BigEndian();
    long readInt64_BigEndian();

    byte readByte();
    QByteArray readBytes(const int length);

    double readDouble();
    float readFloat();
    double readDouble_BigEndian();
    float readFloat_BigEndian();

    int Position();
    void seek(int position);
    void jump(int offset);

    static QByteArray toByteArray(qint32 integer);
};

#endif // BIGENDIANREADER_H
