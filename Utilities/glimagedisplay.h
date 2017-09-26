#ifndef GLIMAGEDISPLAY_H
#define GLIMAGEDISPLAY_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>

#include <QVector>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)

class GLImageDisplay : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

private:
    QOpenGLShaderProgram *m_program;

    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_buffer;
    QVector<GLfloat> m_data;

    int m_projMatrixLoc;
    int m_mvMatrixLoc;
    int m_normalMatrixLoc;
    int m_lightPosLoc;
    QMatrix4x4 m_proj;
    QMatrix4x4 m_camera;
    QMatrix4x4 m_world;


public:
    GLImageDisplay(QWidget *parent = 0);

    // QOpenGLWidget interface
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
};

#endif // GLIMAGEDISPLAY_H
