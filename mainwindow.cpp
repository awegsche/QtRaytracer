#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QString>
#include "camera.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    ui->label->setPixmap(QPixmap::fromImage(_image));

    _world = new World();
    _world->build();


    _image = QImage(_world->vp.hres, _world->vp.vres, QImage::Format_RGB32);
    _image.fill(0xA0FFFF);
    _image.setPixel(10, 10, 0xFF0000);

    i_height = _world->vp.vres;

    connect(_world, SIGNAL(display_pixel(int,int,int, int, int)), this, SLOT(display_pixel(int,int,int,int,int)));
    connect(_world, SIGNAL(done()), this, SLOT(done()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    _image.fill(0xA0FFFF);
    clock.start();
    _world->start();
}

void MainWindow::display_pixel(int x, int y, int r, int g, int b)
{
//    uint rgb = r ;
//    rgb |= g << 8;
//    rgb |= b << 16;
    uint rgb = r << 16 | g << 8 | b;
    _image.setPixel(y, i_height - x, rgb);
    //this->setWindowTitle(QString::number(x) + ", " + QString::number(y));
    if (clock.elapsed() > 10) {
        ui->label->setPixmap(QPixmap::fromImage(_image));
        clock.restart();
    }

    //ui->label->setPixmap(QPixmap::fromImage(_image));
}

void MainWindow::done()
{
    ui->label->setPixmap(QPixmap::fromImage(_image));
    this->setWindowTitle("DOne");
}
