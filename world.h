#ifndef WORLD_H
#define WORLD_H


#include "rgbcolor.h"
#include "sphere.h"
#include "viewplane.h"
#include "tracer.h"
#include "qthread.h"
#include "ray.h"
#include "geometricobject.h"
#include "light.h"
#include <QThread>
#include "pixel.h"
#include "mcworld.h"
#include "grid.h"
#include "textureholder.h"

#include <vector>

class Camera;

class World : public QThread
{
    Q_OBJECT
public:
    //QMutex mutex;
    ViewPlane vp;
    RGBColor background_color;
    std::vector<GeometricObject*> objects;
    Tracer* tracer_ptr;
    Camera* camera_ptr;
    Light* ambient_ptr;
    std::vector<Light*> lights;
    bool running;
    Grid* world_grid;
    TextureHolder* tholder;

public:
    World(QObject* parent = nullptr);

    void build();
    void load_MC(MCWorld* w);
    void add_object(GeometricObject* o);
    void add_light(Light* l);
    void add_chunks(MCWorld* world, int x, int y);

    void render_scene_();
    void render_camera();
    ShadeRec hit_bare_bones_objects(const Ray &ray);
    ShadeRec hit_objects(const Ray& ray);
    void dosplay_p(int r, int c, const RGBColor &pixel_color);
    //static Pixel display_p(Pixel& result, const Pixel& p);

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
