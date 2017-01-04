#include "imagedisplay.h"
#include <QPainter>
#include <QDebug>

ImageDisplay::ImageDisplay(QWidget *parent) : QWidget(parent) {
  m_image = 0;
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void ImageDisplay::setImage(QImage *image) {
  m_image = image;

  //setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  qDebug() << m_image->size().height() << ", " ;
  repaint();


 // setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
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
