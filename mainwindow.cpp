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


void MainWindow::loadchunk(int y_, int x_)
{
    QString path_ = QString("C:/Users/Andreas.DESKTOP-D87O57E/AppData/Roaming/.minecraft/saves/Alkas/region/r.%1.%2.mca")
            .arg(QString::number(y_)).arg(QString::number(x_));

    NBTFileReader F(path_);
    MCWorld* W = new MCWorld();
    F.Load(W);

    _world->add_chunks(W, y_, x_);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);




    //ui->label->setPixmap(QPixmap::fromImage(_image));
    m_downsampling = 3;
    i_downsampling = m_downsampling;

    _world = new PreviewWorld(i_downsampling, 8);

    _world->build();


    loadchunk(0, 0);
  /*  loadchunk(0, -1);
    loadchunk(0, -2);
    loadchunk(0, -3);
    loadchunk(1, 0);
    loadchunk(-1, 0);
    loadchunk(1, -1);
    loadchunk(-1, -1);
    loadchunk(1, -2);
    loadchunk(1, -3);*/
    //_world->world_grid->setup_cells();
    //_world->add_object(_world->world_grid);
    //ui->treeView->setModel(W);



    _image = QImage(_world->vp.hres, _world->vp.vres, QImage::Format_RGB32);
    _image.fill(0xA0FFFF);
    _image.setPixel(10, 10, 0xFF0000);

    i_height = _world->vp.vres;


    _display = new ImageDisplay(this);
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
    if (clock.elapsed() > 33) {
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

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    qDebug() << QString("%1 pressed").arg(event->key());

    last_line = 0;
    i_downsampling = m_downsampling;
    if (event->key() == Qt::Key_Space)
    {
        _world->preview = false;

        i_downsampling = 1;
    }
    else
        _world->Keypressed(event->key());
    clock.start();
    clock2.start();
    if (_world->isRunning()){
       _world->running = false;
       _world->wait();
    }
    _world->start();
}

