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

#ifdef WIN64 | WIN32
    const QString STR_REGIONSPATH = QString("C:/Users/Andreas.DESKTOP-D87O57E/AppData/Roaming/.minecraft/saves/Alkas/region");
#else
    const QString STR_REGIONSPATH = QString("/home/awegsche/Dropbox/Minecraft Save");
#endif

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

    void loadchunk(const QString& path, int y_, int x_);
    
public slots:
    void display_pixel(int x, int y, int r, int g, int b);
    void done();

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event);
};

void checkPaths();

#endif // MAINWINDOW_H
