#include "imagedisplay.h"
#include <QPainter>
#include <QDebug>
#include "mainwindow.h"
#include <QKeyEvent>
#include "thinlens.h"
#include <QMenu>

ImageDisplay::ImageDisplay(MCSceneRenderer *w, MainWindow *mainw, QWidget *parent) : QWidget(parent) {
  m_image = 0;
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  setFocusPolicy(Qt::ClickFocus);
  setMaximumSize(1280, 640);
  mw = w;
  mainwindow = mainw;

}

void ImageDisplay::setImage(QImage *image) {
  m_image = image;

  //setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
//  qDebug() << m_image->size().width() << ", " << m_image->size().width();
  repaint();
    setContextMenuPolicy(Qt::CustomContextMenu);

  // setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void ImageDisplay::save_image(const QString &filename) const
{
    m_image->save(filename, "jpeg", 80);
}

void ImageDisplay::showContextMenu(const QPointF &p)
{
    QMenu contextMenu(tr("title"), this);
    contextMenu.addSection(tr("section 1"));

    contextMenu.setFixedSize(100,100);
    contextMenu.popup(mapToGlobal(QPoint((int)p.x(), (int)p.y())));
}

void ImageDisplay::paintEvent(QPaintEvent*) {
  if (!m_image) { return; }
  QPainter painter(this);
//  qDebug() << m_image->size().width() << ", " << m_image->size().width();
  painter.drawImage(rect(), *m_image, m_image->rect());
}

QSize ImageDisplay::sizeHint() const
{
    if (m_image)
        return m_image->size();
    return QSize(640, 480);
}

void ImageDisplay::keyPressEvent(QKeyEvent *event)
{
//    QKeyEvent *event = static_cast<QKeyEvent *>(e);
//    qDebug() << QString("%1 pressed").arg(event->key());

    mainwindow->stoprender();

    if (event->key() == Qt::Key_Space)
    {
        mw->switch_to_render();

    }
	else
	{
        mw->switch_to_preview();
        mw->Keypressed(event->key());

	}
    mainwindow->render();
}

QSize ImageDisplay::minimumSizeHint() const
{
    return QSize(32, 32);
}

void ImageDisplay::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        showContextMenu(event->localPos());
    }
}


