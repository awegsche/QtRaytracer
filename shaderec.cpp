#include "shaderec.h"

#include "point.h"
#include "vector.h"
#include "rgbcolor.h"


ShadeRec::ShadeRec( World *world)
    :   w(world),
        hit_an_object(false),
        local_hit_point(),
        normal(),
        color()
{

}
