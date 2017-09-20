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

#ifdef WIN64
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

    QTime clock;
    QTime clock2;

public:
    PreviewWorld* _world;

    int i_width;
    int i_height;
    int i_downsampling;
    int m_downsampling;
    double _aperture;

    // for updating the pixmap:
    int last_line;
    ImageDisplay *_display;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void render();

private slots:
    void on_pushButton_clicked();

    void on_camPosX_editingFinished();

    void on_camPosY_editingFinished();

    void on_camPosZ_editingFinished();

    void on_distanceSlider_sliderMoved(int position);

    void on_supersamplingBox_editingFinished();

    void on_focusSlider_sliderReleased();

    void on_distanceSlider_sliderReleased();

private:
    Ui::MainWindow *ui;

    void loadchunk(const QString& path, int y_, int x_);
    void update_camera_info();
    
public slots:
    void display_pixel(int x, int y, int r, int g, int b);
    void done();

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event);
};

void checkPaths();

#endif // MAINWINDOW_H
