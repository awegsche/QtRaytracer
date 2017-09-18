#ifndef NOSHADEMATTE_H
#define NOSHADEMATTE_H

#include "matte.h"

class NoShadeMatte : public Matte
{
public:
    NoShadeMatte();
    NoShadeMatte(float ka_, float kd_, float r_, float g_, float b_);
    NoShadeMatte(float ka_, float kd_, Texture* t);


    // Material interface
public:
    RGBColor shade(ShadeRec &sr);
};

#endif // NOSHADEMATTE_H
