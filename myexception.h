#ifndef MYEXCEPTION_H
#define MYEXCEPTION_H

#include <QException>
#include <QString>

class MyException : public QException
{
private:
    QString message;
public:
    MyException();
    MyException(const QString& mess);
    MyException(const MyException& exc);

    QString get_message() const;

    // QException interface
public:
    void raise() const;
    QException *clone() const;
};

#endif // MYEXCEPTION_H
