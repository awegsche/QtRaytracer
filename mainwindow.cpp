#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QString>
#include "camera.h"
#include "imagedisplay.h"
#include <QPainter>
#include <QKeyEvent>
#include <QDebug>
#include "mcworld.h"
#include "nbtfilereader.h"
#include <QDir>
#include "pinhole.h"
#include <QApplication>


void MainWindow::loadchunk(const QString& path, int y_, int x_)
{
    QString path_ =  path + QDir::separator() + QString("r.%1.%2.mca")
            .arg(QString::number(y_)).arg(QString::number(x_));

    NBTFileReader F(path_);
    MCWorld* W = new MCWorld();
    F.Load(W);

    _world->add_chunks(W, y_, x_);
}

void MainWindow::update_camera_info()
{
    ui->camPosX->setValue(_world->camera_ptr->eye.X);
    ui->camPosY->setValue(_world->camera_ptr->eye.Y);
    ui->camPosZ->setValue(_world->camera_ptr->eye.Z);
    ui->camDirX->setValue(_world->camera_ptr->v.X);
    ui->camDirY->setValue(_world->camera_ptr->v.Y);
    ui->camDirZ->setValue(_world->camera_ptr->v.Z);

    double d = ((Pinhole*)_world->camera_ptr)->get_distance();
    ui->distanceSlider->setValue((int)d);
    ui->distanceValue->setText(QString::number(d, 'g', 1));
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);



    //ui->label->setPixmap(QPixmap::fromImage(_image));
    m_downsampling = 3;
    i_downsampling = m_downsampling;

    _world = new PreviewWorld(i_downsampling, 4);

    _world->build();


    loadchunk(STR_REGIONSPATH, 0, 0);
  /*  loadchunk(STR_REGIONSPATH, 0, -1);
    loadchunk(STR_REGIONSPATH, 0, -2);
    loadchunk(STR_REGIONSPATH, 0, -3);
    loadchunk(STR_REGIONSPATH, 1, 0);
    loadchunk(STR_REGIONSPATH, -1, 0);
    loadchunk(STR_REGIONSPATH, 1, -1);
    loadchunk(STR_REGIONSPATH, -1, -1);
    loadchunk(STR_REGIONSPATH, 1, -2);
    loadchunk(STR_REGIONSPATH, 1, -3);*/
    //_world->world_grid->setup_cells();
    //_world->add_object(_world->world_grid);
    //ui->treeView->setModel(W);



    _image = QImage(_world->vp.hres, _world->vp.vres, QImage::Format_RGB32);
    _image.fill(0xA0FFFF);
    _image.setPixel(10, 10, 0xFF0000);

    i_height = _world->vp.vres;


    _display = new ImageDisplay(this, this);

    _display->setImage(&_image);
    ((QVBoxLayout*)ui->frame->layout())->insertWidget(0, _display);

    connect(_world, SIGNAL(display_pixel(int,int,int, int, int)), this, SLOT(display_pixel(int,int,int,int,int)));
    connect(_world, SIGNAL(done()), this, SLOT(done()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
   // _image.fill(0xA0FFFF);
    last_line = 0;

    clock.start();
    clock2.start();
    _world->preview = false;

    i_downsampling = 1;
    _world->start();
}

void MainWindow::display_pixel(int x, int y, int r, int g, int b)
{
//    uint rgb = r ;
//    rgb |= g << 8;
//    rgb |= b << 16;
    uint rgb = r << 16 | g << 8 | b;

    for (int i = 0; i < i_downsampling; i++)
        for (int j = 0; j < i_downsampling; j++)
            _image.setPixel(y * i_downsampling + j, (i_height - x * i_downsampling) - i - 1, rgb);
    //this->setWindowTitle(QString::number(x) + ", " + QString::number(y));
    if (clock.elapsed() > 50) {
        _display->repaint();
        //ui->label->setPixmap(QPixmap::fromImage(_image));
        clock.restart();
        ui->label_info->setText(QString("%1 s").arg((float)clock2.elapsed()/1000.0f));
    }

    //ui->label->setPixmap(QPixmap::fromImage(_image));
}

void MainWindow::done()
{
    //ui->label->setPixmap(QPixmap::fromImage(_image));
    _display->repaint();
    this->setWindowTitle("Done");
    float elapsed = (float)clock2.elapsed();
    ui->label_info->setText(QString("Rendering took %1 s\n%2 Pixels per second")
                            .arg(elapsed/1000.0f)
                            .arg((float)(_world->vp.hres * _world->vp.vres) / elapsed * 1000.0));
}

void MainWindow::render()
{
    clock.start();
    clock2.start();
    if (_world->isRunning()){
       _world->running = false;
       _world->wait();
    }
    update_camera_info();
    _world->start();
}


void MainWindow::keyPressEvent(QKeyEvent *event)
{

}


void checkPaths()
{

}

void MainWindow::on_camPosX_editingFinished()
{
    this->_world->camera_ptr->eye.X = ui->camPosX->value();
    render();
}

void MainWindow::on_camPosY_editingFinished()
{
    this->_world->camera_ptr->eye.Y = ui->camPosY->value();
    render();
}

void MainWindow::on_camPosZ_editingFinished()
{
    this->_world->camera_ptr->eye.Z = ui->camPosZ->value();
    render();
}

void MainWindow::on_distanceSlider_sliderMoved(int position)
{
    Pinhole *p = static_cast<Pinhole*>(_world->camera_ptr);
    p->set_distance((double) position);
    render();
}

void MainWindow::on_supersamplingBox_editingFinished()
{
    _world->set_sampler(ui->supersamplingBox->value());
}
