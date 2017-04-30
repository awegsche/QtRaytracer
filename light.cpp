#include "light.h"

Light::Light()
    : shadows(true)
{

}

bool Light::casts_shadows()
{
    return shadows;
}

void Light::set_shadows(bool b)
{
    shadows = b;
}
