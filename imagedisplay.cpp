#include "imagedisplay.h"
#include <QPainter>
#include <QDebug>
#include "mainwindow.h"
#include <QKeyEvent>
#include "thinlens.h"

ImageDisplay::ImageDisplay(MainWindow *w, QWidget *parent) : QWidget(parent) {
  m_image = 0;
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  setFocusPolicy(Qt::ClickFocus);
  mw = w;
}

void ImageDisplay::setImage(QImage *image) {
  m_image = image;

  //setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  qDebug() << m_image->size().height() << ", " ;
  repaint();


  // setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void ImageDisplay::save_image(const QString &filename) const
{
    m_image->save(filename, "jpeg", 80);
}

void ImageDisplay::paintEvent(QPaintEvent*) {
  if (!m_image) { return; }
  QPainter painter(this);
  painter.drawImage(rect(), *m_image, m_image->rect());
}

QSize ImageDisplay::sizeHint() const
{
    return m_image->size();
}

void ImageDisplay::keyPressEvent(QKeyEvent *event)
{
//    QKeyEvent *event = static_cast<QKeyEvent *>(e);
    qDebug() << QString("%1 pressed").arg(event->key());

    ThinLens* camera = static_cast<ThinLens*>(mw->_world->camera_ptr);

    mw->last_line = 0;
    mw->i_downsampling = mw->m_downsampling;
    camera->_aperture = 0.0;
    if (event->key() == Qt::Key_Space)
    {
        mw->_world->preview = false;
        camera->_aperture = mw->_aperture;
        mw->i_downsampling = 1;
    }
    else
        mw->_world->Keypressed(event->key());
    mw->render();
}
