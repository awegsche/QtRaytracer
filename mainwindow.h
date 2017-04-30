#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QImage>
#include <QLabel>
#include "world.h"
#include "rgbcolor.h"
#include <QTime>
#include "imagedisplay.h"
#include "previewworld.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QPixmap _pixmap;
    QImage _image;
    PreviewWorld* _world;

    QTime clock;
    QTime clock2;
    int i_width;
    int i_height;
    int i_downsampling;
    int m_downsampling;

    // for updating the pixmap:
    int last_line;
    ImageDisplay *_display;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;

public slots:
    void display_pixel(int x, int y, int r, int g, int b);
    void done();

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event);
};

#endif // MAINWINDOW_H
