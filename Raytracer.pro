
QT       += core gui
CONFIG += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


INCLUDEPATH += $$PWD/BRDFs
INCLUDEPATH += $$PWD/Cameras
INCLUDEPATH += $$PWD/GeometricObjects
INCLUDEPATH += $$PWD/GUI
INCLUDEPATH += $$PWD/Lights
INCLUDEPATH += $$PWD/Materials
INCLUDEPATH += $$PWD/Maths
INCLUDEPATH += $$PWD/NBT
INCLUDEPATH += $$PWD/Samplers
INCLUDEPATH += $$PWD/Textures
INCLUDEPATH += $$PWD/Tracers
INCLUDEPATH += $$PWD/Utilities
INCLUDEPATH += $$PWD/World


TARGET = Raytracer
TEMPLATE = app

HEADERS += \
    BRDFs/brdf.h \
    BRDFs/glossyspecular.h \
    BRDFs/lambertian.h \
    BRDFs/perfectspecular.h \
    Cameras/camera.h \
    Cameras/pinhole.h \
    Cameras/thinlens.h \
    GeometricObjects/bbox.h \
    GeometricObjects/compound.h \
    GeometricObjects/FlatMeshTriangle.h \
    GeometricObjects/geometricobject.h \
    GeometricObjects/grid.h \
    GeometricObjects/mcblock.h \
    GeometricObjects/mcgrid.h \
    GeometricObjects/mcisoblock.h \
    GeometricObjects/mcregiongrid.h \
    GeometricObjects/mcstandardblock.h \
    GeometricObjects/mcwaterblock.h \
    GeometricObjects/mesh.h \
    GeometricObjects/meshtriangle.h \
    GeometricObjects/plane.h \
    GeometricObjects/SmoothMeshTriangle.h \
    GeometricObjects/sphere.h \
    GeometricObjects/Triangle.h \
    GUI/mainwindow.h \
    GUI/qrendercanvas.h \
    Lights/ambient.h \
    Lights/ambientoccluder.h \
    Lights/light.h \
    Lights/pointlight.h \
    Materials/material.h \
    Materials/matte.h \
    Materials/noshadematte.h \
    Materials/phong.h \
    Materials/reflective.h \
    Maths/matrix.h \
    Maths/normal.h \
    Maths/point.h \
    Maths/point2d.h \
    Maths/ray.h \
    Maths/rgbcolor.h \
    Maths/vector.h \
    NBT/bigendianreader.h \
    NBT/chunk.h \
    NBT/littleendianreader.h \
    NBT/nbtfilereader.h \
    NBT/nbttag.h \
    NBT/nbttagbyte.h \
    NBT/nbttagbytearray.h \
    NBT/nbttagcompound.h \
    NBT/nbttagdouble.h \
    NBT/nbttagend.h \
    NBT/nbttagfloat.h \
    NBT/nbttagint.h \
    NBT/nbttagintarray.h \
    NBT/nbttaglist.h \
    NBT/nbttaglong.h \
    NBT/nbttagstring.h \
    Samplers/Jittered.h \
    Samplers/MultiJittered.h \
    Samplers/NRooks.h \
    Samplers/pseudorandom.h \
    Samplers/PureRandom.h \
    Samplers/sampler.h \
    Textures/constantcolor.h \
    Textures/imagetexture.h \
    Textures/texture.h \
    Textures/textureholder.h \
    Tracers/multipleobjects.h \
    Tracers/raycast.h \
    Tracers/tracer.h \
    Utilities/constants.h \
    Utilities/glimagedisplay.h \
    Utilities/imagedisplay.h \
    Utilities/myexception.h \
    Utilities/obj.h \
    Utilities/pixel.h \
    World/mcworld.h \
    World/previewworld.h \
    World/shaderec.h \
    World/viewplane.h \
    World/world.h \
    World/mcscenerenderer.h

SOURCES += \
    main.cpp \
    BRDFs/brdf.cpp \
    BRDFs/glossyspecular.cpp \
    BRDFs/lambertian.cpp \
    BRDFs/perfectspecular.cpp \
    Cameras/camera.cpp \
    Cameras/pinhole.cpp \
    Cameras/thinlens.cpp \
    GeometricObjects/bbox.cpp \
    GeometricObjects/compound.cpp \
    GeometricObjects/FlatMeshTriangle.cpp \
    GeometricObjects/geometricobject.cpp \
    GeometricObjects/grid.cpp \
    GeometricObjects/mcblock.cpp \
    GeometricObjects/mcgrid.cpp \
    GeometricObjects/mcisoblock.cpp \
    GeometricObjects/mcregiongrid.cpp \
    GeometricObjects/mcstandardblock.cpp \
    GeometricObjects/mcwaterblock.cpp \
    GeometricObjects/mesh.cpp \
    GeometricObjects/meshtriangle.cpp \
    GeometricObjects/plane.cpp \
    GeometricObjects/SmoothMeshTriangle.cpp \
    GeometricObjects/sphere.cpp \
    GeometricObjects/Triangle.cpp \
    GUI/mainwindow.cpp \
    GUI/qrendercanvas.cpp \
    Lights/ambient.cpp \
    Lights/ambientoccluder.cpp \
    Lights/light.cpp \
    Lights/pointlight.cpp \
    Materials/material.cpp \
    Materials/matte.cpp \
    Materials/noshadematte.cpp \
    Materials/phong.cpp \
    Materials/reflective.cpp \
    Maths/matrix.cpp \
    Maths/normal.cpp \
    Maths/point.cpp \
    Maths/point2d.cpp \
    Maths/ray.cpp \
    Maths/rgbcolor.cpp \
    Maths/vector.cpp \
    NBT/bigendianreader.cpp \
    NBT/chunk.cpp \
    NBT/littleendianreader.cpp \
    NBT/nbtfilereader.cpp \
    NBT/nbttag.cpp \
    NBT/nbttagbyte.cpp \
    NBT/nbttagbytearray.cpp \
    NBT/nbttagcompound.cpp \
    NBT/nbttagdouble.cpp \
    NBT/nbttagend.cpp \
    NBT/nbttagfloat.cpp \
    NBT/nbttagint.cpp \
    NBT/nbttagintarray.cpp \
    NBT/nbttaglist.cpp \
    NBT/nbttaglong.cpp \
    NBT/nbttagstring.cpp \
    Samplers/Jittered.cpp \
    Samplers/MultiJittered.cpp \
    Samplers/NRooks.cpp \
    Samplers/pseudorandom.cpp \
    Samplers/PureRandom.cpp \
    Samplers/sampler.cpp \
    Textures/constantcolor.cpp \
    Textures/imagetexture.cpp \
    Textures/texture.cpp \
    Textures/textureholder.cpp \
    Tracers/multipleobjects.cpp \
    Tracers/raycast.cpp \
    Tracers/tracer.cpp \
    Utilities/glimagedisplay.cpp \
    Utilities/imagedisplay.cpp \
    Utilities/myexception.cpp \
    Utilities/obj.cpp \
    Utilities/pixel.cpp \
    World/build_cubes.cpp \
    World/mcworld.cpp \
    World/previewworld.cpp \
    World/shaderec.cpp \
    World/viewplane.cpp \
    World/world.cpp \
    World/mcscenerenderer.cpp

FORMS += \
    GUI/mainwindow.ui

