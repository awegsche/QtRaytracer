#ifndef WORLD_H
#define WORLD_H

#include "rgbcolor.h"
#include "sphere.h"
#include "viewplane.h"
#include "tracer.h"
#include "qthread.h"
#include "ray.h"
#include "geometricobject.h"
#include <QThread>

#include <vector>

class World : public QThread
{
    Q_OBJECT
public:
    //QMutex mutex;
    ViewPlane vp;
    RGBColor background_color;
    std::vector<GeometricObject*> objects;
    Tracer* tracer_ptr;

public:
    World(QObject* parent = nullptr);

    void build();
    void add_object(GeometricObject* o);

    void render_scene_();
    ShadeRec hit_bare_bones_objects(const Ray &ray);

   // void open_window(const int hres, const int vres) const;
    //void display(const int row, const int column, const RGBColor& pixel_color) const;

signals:
    void display_pixel(int row, int column, int r, int g, int b);
    void done();

    // QThread interface
protected:
    void run();
};

#endif // WORLD_H
