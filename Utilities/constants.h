#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <vectorclass.h>
#include "vector3d.h"

typedef Vec3f real3;
typedef float real;


typedef unsigned char byte;

///<summary>
/// Everything smaller than kEpsilon will be considered zero.
/// Change kEpsilon to a smaller value (maybe 1.0e-7 for double).
///</summary>
const real kEpsilon = 1.0e-3; 

const real kHugeValue = 1.0e+12;
const real Pi = 3.141592653589793238462643383279502;
const real invPi = 1.0 / Pi;
const real TWO_PI = 2 * Pi;
const real GRAD = Pi / 180.0;

// The edge length of one Minecraft block. If 1.0 the games coordinates will be equal to the Raytracer's world coordinates
const real BLOCKLENGTH = 1.0;





#endif // CONSTANTS_H
