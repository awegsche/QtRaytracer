#ifndef AMBIENT_H
#define AMBIENT_H

#include "light.h"

class Ambient : public Light
{
private:
    float ls;
    RGBColor color;

public:
    Ambient();

    // Light interface
public:
    Vector get_direction(ShadeRec &sr);
    RGBColor L(ShadeRec &sr);
};

#endif // AMBIENT_H
