#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include "mcscenerenderer.h"

#include <QObject>
#include <QWidget>

class MainWindow;

class ImageDisplay : public QWidget
{
    Q_OBJECT
public:
    explicit ImageDisplay(MCSceneRenderer *w, MainWindow* mainw, QWidget *parent = 0);
    void setImage(QImage *image);
    void save_image(const QString& filename) const;

    MCSceneRenderer* mw;
    MainWindow* mainwindow;

signals:

public slots:
    void showContextMenu(const QPointF &p);

    // QWidget interface
protected:
    void paintEvent(QPaintEvent *event);

private:
    QImage* m_image;

    // QWidget interface
public:
    QSize sizeHint() const;

    // QWidget interface
protected:
    void keyPressEvent(QKeyEvent *event) Q_DECL_OVERRIDE;

    // QWidget interface
public:
    QSize minimumSizeHint() const Q_DECL_OVERRIDE;



    // QWidget interface
protected:
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
};

#endif // IMAGEDISPLAY_H
