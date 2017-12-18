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
#include "thinlens.h"
#include "PureRandom.h"
#include <QFileDialog>
#include <qfileinfo.h>

const real HAZE_STEP = 50.0;
const real HAZE_ATT_STEP = 1.0e6;
const real DISTANCE_STEP = 10.0;
const real ANGLE_STEP = 125.0;
const real APERTURE_STEP = 1000.0;

void MainWindow::loadchunk(const QString& path, int y_, int x_)
{
    QString path_ =  path + QDir::separator() + QString("r.%1.%2.mca")
            .arg(QString::number(y_)).arg(QString::number(x_));
	QFileInfo fi(path);
	if (fi.exists())
	{
		NBTFileReader F(path_);
		MCWorld* W = new MCWorld();
		F.Load(W);

		_world->add_chunks(W, y_, x_);
	}
}

void MainWindow::update_camera_info()
{
    ui->camPosX->setValue(_world->camera_ptr->eye.X());
    ui->camPosY->setValue(_world->camera_ptr->eye.Y());
    ui->camPosZ->setValue(_world->camera_ptr->eye.Z());
    ui->camDirX->setValue(_world->camera_ptr->u.X());
    ui->camDirY->setValue(_world->camera_ptr->u.Y());
    ui->camDirZ->setValue(_world->camera_ptr->u.Z());

//    double d = ((Pinhole*)_world->camera_ptr)->get_zoom();
//    ui->distanceSlider->setValue((int)(d*10));
//    ui->distanceValue->setText(QString::number(d, 'f', 1));
}

void MainWindow::resize_display()
{
	i_height = _world->vp.vres;
	i_width = _world->vp.hres;

    _image = QImage(_world->vp.hres, _world->vp.vres, QImage::Format_RGB32);
    _image.fill(0xA0FFFF);
    _image.setPixel(10, 10, 0xFF0000);

    _display->setImage(&_image);

    int wmax = i_width < 1024 ? i_width : 1024;
    int hmax = i_height < 768 ? i_height : 768;
    _display->setFixedSize(wmax, hmax);
    _display->adjustSize();
    //ui->frame->adjustSize();
    //resize(minimumSizeHint());
//    ui->centralWidget->layout()->

}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    i_width(640), i_height(480)
{
    ui->setupUi(this);

    _world = new MCSceneRenderer();

    _world->build();


    ui->angleSlider->setValue(_world->get_angle() * ANGLE_STEP);
    ui->angleValue->setText(QString::number(_world->get_angle()));
    ui->distanceSlider->setValue(_world->get_vp_distance() * DISTANCE_STEP);
    ui->distanceValue->setText(QString::number(_world->get_vp_distance()));
    ui->apertureValue->setText("1.0");
    ui->hazeattenuationSlider->setValue(_world->haze_attenuation * HAZE_ATT_STEP);
    ui->hazeSlider->setValue(_world->haze_distance * HAZE_STEP);
    ui->checkBox->setChecked(_world->haze);
    ui->dial->setValue(500);

	


//#if defined NDEBUG || defined QT_NO_DEBUG
//	for (int i = -5; i < 6; i++)
//		for (int j = -5; j < 6; j++)
//			if(!(j==-1 && i ==0))
//				loadchunk(STR_REGIONSPATH, i, j);
//#else
	
//    loadchunk(STR_REGIONSPATH, 0, 0);
//    ////loadchunk(STR_REGIONSPATH, 0, -1);
//    //loadchunk(STR_REGIONSPATH, 0, -2);
//    //loadchunk(STR_REGIONSPATH, 0, -3);

//#endif // NDEBUG
//    //_world->world_grid->setup_cells();
//    //_world->add_object(_world->world_grid);
//    //ui->treeView->setModel(W);

//    _world->resize_vp(640, 480);

    _display = new ImageDisplay(_world, this);

    ((QVBoxLayout*)ui->frame->layout())->insertWidget(0, _display);
    qDebug() <<((QVBoxLayout*)ui->frame->layout())->setAlignment(_display, Qt::AlignHCenter);

    resize_display();

//    connect(_world, SIGNAL(display_pixel(int,int,int, int, int)), this, SLOT(display_pixel(int,int,int,int,int)));
//    connect(_world, SIGNAL(display_line(int,const uint*)), this, SLOT(display_line(int,const uint*)));
//    connect(_world, SIGNAL(done()), this, SLOT(done()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("Open Image"), QString(), tr("Image Files (*.jpg)"));

    if (fileName != QString())
        _display->save_image(fileName);
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

void MainWindow::display_line(const int line, const uint *rgb)
{
    for (int x = 0; x < i_width  ; x++)
                _image.setPixel(x, i_height - line - 1, rgb[x]);

    if (clock.elapsed() > 50) {
        _display->repaint();
        //ui->label->setPixmap(QPixmap::fromImage(_image));
        clock.restart();
        ui->label_info->setText(QString("%1 s").arg((float)clock2.elapsed()/1000.0f));
    }

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
    update_camera_info();
    _world->start();
}

void MainWindow::stoprender()
{
    if (_world->isRunning()){
       _world->running = false;
       _world->wait();
    }
}


void MainWindow::keyPressEvent(QKeyEvent *event)
{

}


void checkPaths()
{

}

void MainWindow::on_camPosX_editingFinished()
{
    stoprender();

    this->_world->camera_ptr->eye.data.insert(0, ui->camPosX->value());
    render();
}

void MainWindow::on_camPosY_editingFinished()
{
    stoprender();
    this->_world->camera_ptr->eye.data.insert(1, ui->camPosY->value());
    render();
}

void MainWindow::on_camPosZ_editingFinished()
{
    this->_world->camera_ptr->eye.data.insert(2,  ui->camPosZ->value());
    render();
}

void MainWindow::on_distanceSlider_sliderMoved(int position)
{
    double dist = (double) ui->distanceSlider->value() / DISTANCE_STEP;
    double angle = (double) ui->angleSlider->value() / ANGLE_STEP;



    ui->angleValue->setText(QString::number(angle, 'f', 1));
    ui->distanceValue->setText(QString::number(dist, 'f', 1));
}

void MainWindow::on_supersamplingBox_editingFinished()
{
    stoprender();
    _world->set_samples(ui->supersamplingBox->value());
    
}

void MainWindow::on_focusSlider_sliderReleased()
{

//    double dist = (double) ui->distanceSlider->value() / DISTANCE_STEP;
//    double angle = (double) ui->angleSlider->value() / ANGLE_STEP;


//    _world->set_vp_distance(dist, angle);

//    ui->angleValue->setText(QString::number(angle, 'f', 1));
//    ui->distanceValue->setText(QString::number(dist, 'f', 1));
//    //i_downsampling = m_downsampling;
    render();

}

void MainWindow::on_distanceSlider_sliderReleased()
{
    stoprender();
    double dist = (double) ui->distanceSlider->value() / DISTANCE_STEP;
    double angle = (double) ui->angleSlider->value() / ANGLE_STEP;


    _world->set_vp_distance(dist, angle);

    ui->angleValue->setText(QString::number(angle, 'f', 1));
    ui->distanceValue->setText(QString::number(dist, 'f', 1));

    render();

}

void MainWindow::on_dial_sliderReleased()
{
    stoprender();
    double ap = (double)ui->dial->value() / APERTURE_STEP;
	_world->set_aperture(ap);
    ui->apertureValue->setText(QString::number(ap, 'f', 2));
    render();
}


void MainWindow::on_spinBox_height_editingFinished()
{
    _world->resize_vp(ui->spinBox_width->value(), ui->spinBox_height->value());
    resize_display();
}

void MainWindow::on_spinBox_width_editingFinished()
{
    _world->resize_vp(ui->spinBox_width->value(), ui->spinBox_height->value());
    resize_display();

}

void MainWindow::on_checkBox_toggled(bool checked)
{
    _world->haze = checked;
    ui->hazeattenuationSlider->setEnabled(checked);
    ui->hazeattenuationValue->setEnabled(checked);
    ui->hazeSlider->setEnabled(checked);
    ui->hazeValue->setEnabled(checked);

}

void MainWindow::on_hazeSlider_sliderMoved(int position)
{
    real dist = (real)position / HAZE_STEP;
    ui->hazeValue->setText(QString::number(dist, 'f', 1));
}

void MainWindow::on_hazeSlider_sliderReleased()
{
    real dist = (real)ui->hazeSlider->value() / HAZE_STEP;
    ui->hazeValue->setText(QString::number(dist, 'f', 1));
    _world->haze_distance = dist;
    render();
}

void MainWindow::on_hazeattenuationSlider_sliderMoved(int position)
{
    real att = (real)position / HAZE_ATT_STEP;
    ui->hazeattenuationValue->setText(QString::number(att, 'e', 3));
}

void MainWindow::on_hazeattenuationSlider_sliderReleased()
{
    real att = (real)ui->hazeattenuationSlider->value() / HAZE_ATT_STEP;
    ui->hazeattenuationValue->setText(QString::number(att, 'e', 3));
    _world->haze_attenuation = att;
    render();
}

void MainWindow::on_angleSlider_sliderMoved(int position)
{
    double dist = (double) ui->distanceSlider->value() / DISTANCE_STEP;
    double angle = (double) ui->angleSlider->value() / ANGLE_STEP;



    ui->angleValue->setText(QString::number(angle, 'f', 1));
    ui->distanceValue->setText(QString::number(dist, 'f', 1));

}

void MainWindow::on_angleSlider_sliderReleased()
{
    stoprender();

    double dist = (double) ui->distanceSlider->value() / DISTANCE_STEP;
    double angle = (double) ui->angleSlider->value() / ANGLE_STEP;


    _world->set_vp_distance(dist, angle);

    ui->angleValue->setText(QString::number(angle, 'f', 1));
    ui->distanceValue->setText(QString::number(dist, 'f', 1));

    render();
}

void MainWindow::on_actionLoad_regions_triggered()
{
    QFileDialog D;

    D.setFileMode(QFileDialog::Directory);

    QString filename;

    if(D.exec()){
        filename = D.selectedFiles()[0];

        delete _world;

        _world = new MCSceneRenderer();

        //_world->hit_objects_CUDA();

        _world->build();



        QString local_regionspath = filename + QDir::separator() + "region";


    #if defined NDEBUG || defined QT_NO_DEBUG
        for (int i = -5; i < 6; i++)
            for (int j = -5; j < 6; j++)
                loadchunk(local_regionspath, i, j);
    #else

        loadchunk(local_regionspath, 0, 0);
        ////loadchunk(STR_REGIONSPATH, 0, -1);
        //loadchunk(STR_REGIONSPATH, 0, -2);
        //loadchunk(STR_REGIONSPATH, 0, -3);

    #endif // NDEBUG
        //_world->world_grid->setup_cells();
        //_world->add_object(_world->world_grid);
        //ui->treeView->setModel(W);

        _world->resize_vp(640, 480);

        ((QVBoxLayout*)ui->frame->layout())->removeWidget(_display);

        _display = new ImageDisplay(_world, this);


        ((QVBoxLayout*)ui->frame->layout())->insertWidget(0, _display);
        qDebug() <<((QVBoxLayout*)ui->frame->layout())->setAlignment(_display, Qt::AlignHCenter);

        resize_display();

        //    connect(_world, SIGNAL(display_pixel(int,int,int, int, int)), this, SLOT(display_pixel(int,int,int,int,int)));
        connect(_world, SIGNAL(display_line(int,const uint*)), this, SLOT(display_line(int,const uint*)));
        connect(_world, SIGNAL(done()), this, SLOT(done()));

    }


}
