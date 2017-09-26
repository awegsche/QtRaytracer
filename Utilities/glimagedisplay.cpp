#include "glimagedisplay.h"

#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QCoreApplication>
#include <math.h>


static const char *vertexShaderSource =
        "attribute highp vec4 vertex;\n"
         "attribute mediump vec4 texCoord;\n"
         "varying mediump vec4 texc;\n"
         "uniform mediump mat4 matrix;\n"
         "void main(void)\n"
         "{\n"
         "    gl_Position = matrix * vertex;\n"
         "    texc = texCoord;\n"
         "}\n";
static const char *fragmentShaderSource =
        "uniform sampler2D texture;\n"
        "varying mediump vec4 texc;\n"
        "void main(void)\n"
        "{\n"
        "    gl_FragColor = texture2D(texture, texc.st);\n"
        "}\n";



GLImageDisplay::GLImageDisplay(QWidget *parent)
{
    m_data.push_back(0.0f);
    m_data.push_back(0.0f);
    m_data.push_back(0.0f);

    m_data.push_back(1.0f);
    m_data.push_back(0.0f);
    m_data.push_back(0.0f);

    m_data.push_back(0.0f);
    m_data.push_back(1.0f);
    m_data.push_back(0.0f);

    m_data.push_back(0.0f);
    m_data.push_back(1.0f);
    m_data.push_back(0.0f);

    m_data.push_back(0.0f);
    m_data.push_back(1.0f);
    m_data.push_back(0.0f);

    m_data.push_back(1.0f);
    m_data.push_back(1.0f);
    m_data.push_back(0.0f);

}

void GLImageDisplay::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0, 0, 0, 1);

    m_program = new QOpenGLShaderProgram;

    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
    m_program->bindAttributeLocation("vertex", 0);
    m_program->bindAttributeLocation("texCoord", 1);
    m_program->link();

    m_program->bind();
    m_projMatrixLoc = m_program->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_program->uniformLocation("mvMatrix");
    m_normalMatrixLoc = m_program->uniformLocation("normalMatrix");
    m_lightPosLoc = m_program->uniformLocation("lightPos");

    m_vao.create();
    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

    // Setup our vertex buffer object.
    m_buffer.create();
    m_buffer.bind();
    m_buffer.allocate(m_data.constData(), 18 * sizeof(GLfloat));

    // Store the vertex attribute bindings for the program.
    //setupVertexAttribs();

    // Our camera never changes in this example.
    m_camera.setToIdentity();
    m_camera.translate(0, 0, -1);

    // Light position is fixed.
    m_program->setUniformValue(m_lightPosLoc, QVector3D(0, 0, 70));

    m_program->release();
}

void GLImageDisplay::resizeGL(int w, int h)
{

}

void GLImageDisplay::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    m_world.setToIdentity();
   

    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);
    m_program->bind();
    m_program->setUniformValue(m_projMatrixLoc, m_proj);
    m_program->setUniformValue(m_mvMatrixLoc, m_camera * m_world);
    QMatrix3x3 normalMatrix = m_world.normalMatrix();
    m_program->setUniformValue(m_normalMatrixLoc, normalMatrix);



    glDrawArrays(GL_TRIANGLES, 0, 18);

    m_program->release();
}
