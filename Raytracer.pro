#-------------------------------------------------
#
# Project created by QtCreator 2016-12-28T11:18:02
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

#INCLUDEPATH += C:\Qt\Tools\mingw530_32\i686-w64-mingw32\include\c++

TARGET = Raytracer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

#DISTFILES += \
#    Maths.pri
include(Maths.pri)
include(GeometricObjects.pri)
include(Materials.pri)
include(World.pri)
include(Tracers.pri)
include(Cameras.pri)
include(Sampler.pri)
include(BRDFs.pri)
include(Lights.pri)
include(Utilities.pri)

