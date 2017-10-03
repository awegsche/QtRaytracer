#ifndef CAMERA_H
#define CAMERA_H

#include "point.h"
#include "vector.h"
#include "world.h"
#include <QRunnable>

class Camera// : public QRunnable
{
public:
    Point eye;
    Point lookat;
    Vector up;
    Vector u, v, w;
    float exposure_time;

public:
    Camera();
    Camera(const real eye_x, const real eye_y, const real eye_z,
           const real lookat_x, const real lookat_y, const real lookat_z);

    void compute_uvw();
    virtual void render_scene(World &w) = 0;
	virtual void render_scene_CUDA(World& w);

	virtual Ray get_click_ray(const real vpx, const real vpy, const ViewPlane& vp);


    void set_eye(real x, real y, real z);
    void move_eye(real x, real y, real z);
    void move_eye_forward(real d);
    void move_eye_left(real d);
    void move_eye_right(real d);
    void move_eye_backward(real d);
    void rotate_up(real d);
    void rotate_down(real d);
    void rotate_left(real d);
    void rotate_right(real d);
    void set_lookat(real x, real y, real z);
    void set_up(real x, real y, real z);
};

#endif // CAMERA_H
