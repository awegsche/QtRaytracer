#ifndef PINHOLE_H
#define PINHOLE_H

#include "camera.h"
#include "vector"
#include "point2d.h"
#include "concurrentstruct.h"
#include "pixel.h"

class Pinhole : public Camera
{
protected:
    double d;
    double zoom;

public:

    Pixel render_pixel(const ConcurrentStruct &input);

public:
    Pinhole();
    Pinhole(const Camera& cam);
   // Pinhole(const Pinhole& cam);

    Vector ray_direction(const Point2D& p) const;

    // Camera interface
public:
    void render_scene(World &w);

    void set_zoom(double z);
    double get_zoom() const;
    void rescale_zoom(double a);
    void set_distance(double distance);

    double get_distance() const {
        return d;
    }
};

#endif // PINHOLE_H
