#ifndef PINHOLE_H
#define PINHOLE_H

#include "camera.h"
#include "vector"
#include "point2d.h"
#include "concurrentstruct.h"
#include "pixel.h"

class Pinhole : public Camera
{
private:
    float d;
    float zoom;

public:

    Pixel render_pixel(const ConcurrentStruct &input);

public:
    Pinhole();

    Vector ray_direction(const Point2D& p) const;

    // Camera interface
public:
    void render_scene(World &w);

    void set_zoom(float z);
    void set_distance(float distance);
};

#endif // PINHOLE_H
