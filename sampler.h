#ifndef SAMPLER_H
#define SAMPLER_H

#include "point2d.h"

class Sampler
{
public:
    Sampler();

    virtual Point2D sample_unit_square() = 0;
};

#endif // SAMPLER_H
