#include "pseudorandom.h"
#include <cstdlib>

PseudoRandom::PseudoRandom()
{

}

Point2D PseudoRandom::sample_unit_square()
{
    return Point2D((float)rand()/(float)RAND_MAX, (float)rand()/(float)RAND_MAX);
}
