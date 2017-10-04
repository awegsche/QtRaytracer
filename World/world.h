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
#include "mcblock.h"
#include <QMap>
#include "pixel.h"

#include "mcgrid.h"


#include <vector>

class Camera;
class MCRegionGrid;

#ifdef WCUDA
class WorldCUDA {
public:
	GeometricObjectCUDA** objects;
	int num_objects;

	__device__ ShadeRecCUDA hit_objects(const rayCU& ray);

};
#endif // WCUDA


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
    int max_depth;
    bool haze;
    real haze_distance;
    real haze_attenuation;


public:
    World(QObject* parent = nullptr);

    virtual void build() = 0;
    void add_object(GeometricObject* o);
    void add_light(Light* l);

#if defined WCUDA

	MCGridCUDA mcgrid; // for debugging only

	///<summary>
	/// Returns the device pointer to the WorldCUDA objects.
	///</summary>
	WorldCUDA* get_device_world() const;

	///<summary>
	/// Sets up the world with all its objects on the device.
	/// not const because it changes the instance's dev_ptr;
	///</summary>
	WorldCUDA* setup_device_world();

private:
	WorldCUDA* dev_ptr;

#endif
public:
    void render_scene_();
    void render_camera();
    ShadeRec hit_objects(const Ray& ray);
    void dosplay_p(int r, int c, const RGBColor &pixel_color);
    static Pixel display_p(Pixel& result, const Pixel& p);

signals:
    void display_pixel(int row, int column, int r, int g, int b);
    void display_line(const int row, const uint* rgb);
    void done();

    // QThread interface
protected:
    void run();
};



#endif // WORLD_H
