#include "pixel.h"

Pixel::Pixel()
{

}

Pixel::Pixel(const RGBColor &color_, const Point2D &p, World *w_)
    :color(color_), point(p), w(w_){

}
