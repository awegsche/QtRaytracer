#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include "previewworld.h"

#include <QObject>
#include <QWidget>

class MainWindow;

class ImageDisplay : public QWidget
{
    Q_OBJECT
public:
    explicit ImageDisplay(MainWindow *w, QWidget *parent = 0);
    void setImage(QImage *image);


    MainWindow* mw;

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

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event) Q_DECL_OVERRIDE;
};

#endif // IMAGEDISPLAY_H
