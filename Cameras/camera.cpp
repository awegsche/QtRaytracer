#include "camera.h"

Camera::Camera()
    : lookat(0,0,0), up(0,1,0), eye(0,0,-10), exposure_time(1.0f){

}

Camera::Camera(const real eye_x, const real eye_y, const real eye_z, const real lookat_x, const real lookat_y, const real lookat_z)
    :exposure_time(1.0)
{
    set_eye(eye_x, eye_y, eye_z);
    set_lookat(lookat_x, lookat_y, lookat_z);
    set_up(0, 1.0, 0);

    compute_uvw();
}

void Camera::compute_uvw()
{
    w = eye - lookat;
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::render_scene_CUDA(World & w)
{
}

void Camera::set_eye(real x, real y, real z)
{
    eye = Vector(x,y,z);
}

void Camera::move_eye(real x, real y, real z)
{
    eye += Vector(x, y, z);
}

void Camera::move_eye_forward(real d)
{
    eye += w * (-d);
//    lookat += w * (-d);
//    compute_uvw();
}

void Camera::move_eye_left(real d)
{
    eye += u * (-d);
//    lookat += u * (-d);
//    compute_uvw();
}

void Camera::move_eye_right(real d)
{
    eye += u * d;
//    lookat += u * d;
//    compute_uvw();
}

void Camera::move_eye_backward(real d)
{
    eye += w * d;
//    lookat += w * d;
//    compute_uvw();
}

void Camera::rotate_up(real d)
{
    w += up * d;
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::rotate_down(real d)
{
    w += up * (-d);
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::rotate_left(real d)
{
    w += u * d;
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::rotate_right(real d)
{
    w += u * (-d);
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::set_lookat(real x, real y, real z)
{
    lookat = Vector(x,y,z);
}

void Camera::set_up(real x, real y, real z)
{
    up = Vector(x,y,z);
}
