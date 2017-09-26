#include "bigendianreader.h"



BigEndianReader::BigEndianReader(QByteArray &stream)
    : _stream(stream), _position(0)
{

}

short BigEndianReader::readInt16()
{

    int result = 0;
     for(int i = 0; i <2; i++)
        result = (result << 8) + (byte)_stream[_position++];
    return result;
}

int BigEndianReader::readInt24()
{
    int result = 0;
     for(int i = 0; i <3; i++)
        result = (result << 8) + (byte)_stream[_position++];
    return result;
}

int BigEndianReader::readInt32()
{
    int result = 0;
     for(int i = 0; i <4; i++)
        result = (result << 8) + (byte)_stream[_position++];
    return result;
}
long BigEndianReader::readInt64()
{

    for(int i = 0; i <8; i++)
        longu.array[i] = (byte)_stream[_position++];
    return longu.value;
}
short BigEndianReader::readInt16_BigEndian()
{

    for(int i = 1; i >=0; i--)
        shortu.array[i] = (byte)_stream[_position++];
    return shortu.value;
}

int BigEndianReader::readInt24_BigEndian()
{
    int result = 0;
     for(int i = 2; i >= 0; i--)
        result = (result << 8) + (byte)_stream[_position+i];
     _position += 3;
    return result;
}

int BigEndianReader::readInt32_BigEndian()
{

    int result = 0;
     for(int i = 3; i >= 0; i--)
        result = (result << 8) + (byte)_stream[_position+i];
     _position += 4;
    return result;
}
long BigEndianReader::readInt64_BigEndian()
{

    long result = 0;
     for(int i = 7; i >= 0; i--)
        result = (result << 8) + (byte)_stream[_position+i];
     _position += 8;
    return result;
}

//qint16 BigEndianReader::readInt16()
//{
//    qint16 i = 0;
//    i |= static_cast<short>(_stream[_position++]);
//    i |= (short)_stream[_position++] << 8;
//    return i;
//}
//qint32 BigEndianReader::readInt32()
//{
//    qint32 i = 0;
//    i |= (qint32)_stream[_position++];
//    i |= (qint32)_stream[_position++] << 8;
//    i |= (qint32)_stream[_position++] << 16;
//    i |= (qint32)_stream[_position++] << 24;
//    return i;
//}
//qint64 BigEndianReader::readInt64()
//{
//    qint64 i = 0;
//    i |= (qint64)_stream[_position++];
//    i |= (qint64)_stream[_position++] << 8;
//    i |= (qint64)_stream[_position++] << 16;
//    i |= (qint64)_stream[_position++] << 24;
//    i |= (qint64)_stream[_position++] << 32;
//    i |= (qint64)_stream[_position++] << 40;
//    i |= (qint64)_stream[_position++] << 48;
//    i |= (qint64)_stream[_position++] << 56;
//    return i;
//}


byte BigEndianReader::readByte()
{
    return (byte)_stream[_position++];
}

QByteArray BigEndianReader::readBytes(const int length)
{
    QByteArray arr(length, (char)0);
    for(int i = 0; i< length;i++)
        arr[i] = _stream[_position++];
    return arr;
}

double BigEndianReader::readDouble_BigEndian()
{
    for(int i = 7; i >= 0; i--)
        doubleu.array[i] = (byte)_stream[_position++];
    return doubleu.value;

}

float BigEndianReader::readFloat_BigEndian()
{

    for(int i = 3; i >= 0; i--)
        floatu.array[i] = (byte)_stream[_position++];
    return floatu.value;
}
double BigEndianReader::readDouble()
{


     for(int i = 0; i <8; i++)
        doubleu.array[i] = (byte)_stream[_position++];
    return doubleu.value;

}

float BigEndianReader::readFloat()
{

     for(int i = 0; i <4; i++)
        floatu.array[i] = (byte)_stream[_position++];
    return floatu.value;
}

int BigEndianReader::Position()
{
    return _position;
}

void BigEndianReader::seek(int position)
{
    _position = position;
}

void BigEndianReader::jump(int offset)
{
    _position += offset;
}

QByteArray BigEndianReader::toByteArray(qint32 integer)
{
    QByteArray arr(4,0);
    arr[3] = (byte)integer;
    arr[2] = (byte)integer >> 8;
    arr[1] = (byte)integer >> 16;

    arr[0] = (byte)integer >> 24;

    return arr;
}
