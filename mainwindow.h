#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QImage>
#include <QLabel>
#include "world.h"
#include "rgbcolor.h"
#include <QTime>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QPixmap _pixmap;
    QImage _image;
    World* _world;
    QTime clock;

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
};

#endif // MAINWINDOW_H
