#include "ambient.h"

Ambient::Ambient()
    :ls(1.0), color(1.0)
{

}

Vector Ambient::get_direction(ShadeRec &sr)
{
    return Vector();
}

RGBColor Ambient::L(ShadeRec &sr)
{
    return ls * color;
}
