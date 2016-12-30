#include "camera.h"

Camera::Camera()
    : lookat(0,0,0), up(0,1,0), eye(0,0,-10), exposure_time(1.0f){

}

void Camera::compute_uvw()
{
    w = eye - lookat;
    w.normalize();
    u = up ^ w;
    u.normalize();
    v = w ^ u;
}

void Camera::set_eye(real x, real y, real z)
{
    eye = Vector(x,y,z);
}

void Camera::set_lookat(real x, real y, real z)
{
    lookat = Vector(x,y,z);
}

void Camera::set_up(real x, real y, real z)
{
    up = Vector(x,y,z);
}
