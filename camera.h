#ifndef CAMERA_H
#define CAMERA_H

#include "point.h"
#include "vector.h"
#include "world.h"

class Camera
{
protected:
    Point eye;
    Point lookat;
    Vector up;
    Vector u, v, w;
    float exposure_time;

public:
    Camera();

    void compute_uvw();
    virtual void render_scene(World &w) = 0;
    void set_eye(real x, real y, real z);
    void set_lookat(real x, real y, real z);
    void set_up(real x, real y, real z);
};

#endif // CAMERA_H
