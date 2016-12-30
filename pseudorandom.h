#ifndef PSEUDORANDOM_H
#define PSEUDORANDOM_H

#include "sampler.h"

class PseudoRandom : public Sampler
{
public:
    PseudoRandom();

    // Sampler interface
public:
    Point2D sample_unit_square();
};

#endif // PSEUDORANDOM_H
