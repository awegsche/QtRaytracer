#include "myexception.h"

MyException::MyException()
{

}

MyException::MyException(const QString &mess)
    : message(mess){

}

MyException::MyException(const MyException &exc)
    : message(exc.message){

}

QString MyException::get_message() const
{
    return message;
}

void MyException::raise() const
{
    throw *this;
}

QException *MyException::clone() const
{
    return new MyException(*this);
}
