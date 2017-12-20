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
#include "mcscenerenderer.h"

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
    MCSceneRenderer* _world;

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
    void stoprender();

    void resize_display();

private slots:
    void on_pushButton_clicked();

    void on_camPosX_editingFinished();

    void on_camPosY_editingFinished();

    void on_camPosZ_editingFinished();

    void on_distanceSlider_sliderMoved(int position);

    void on_supersamplingBox_editingFinished();

    void on_focusSlider_sliderReleased();

    void on_distanceSlider_sliderReleased();

    void on_dial_sliderReleased();


    void on_spinBox_height_editingFinished();

    void on_spinBox_width_editingFinished();

    void on_checkBox_toggled(bool checked);

    void on_hazeSlider_sliderMoved(int position);

    void on_hazeSlider_sliderReleased();

    void on_hazeattenuationSlider_sliderMoved(int position);

    void on_hazeattenuationSlider_sliderReleased();

    void on_angleSlider_sliderMoved(int position);

    void on_angleSlider_sliderReleased();

    void on_actionLoad_regions_triggered();

    void on_pushButton_2_clicked();

private:
    Ui::MainWindow *ui;

    void loadchunk(const QString& path, int y_, int x_);
    void update_camera_info();

    QString m_bigpicture_path;

    
public slots:
    void display_pixel(int x, int y, int r, int g, int b);
    void display_line(const int line, const uint *rgb);
    void done();
    void big_picture_done();

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event);
};

void checkPaths();

#endif // MAINWINDOW_H
