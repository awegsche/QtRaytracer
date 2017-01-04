#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <QObject>
#include <QWidget>

class ImageDisplay : public QWidget
{
    Q_OBJECT
public:
    explicit ImageDisplay(QWidget *parent = 0);
    void setImage(QImage *image);



signals:

public slots:

    // QWidget interface
protected:
    void paintEvent(QPaintEvent *event);

private:
    QImage* m_image;

    // QWidget interface
public:
    QSize sizeHint() const;
};

#endif // IMAGEDISPLAY_H
