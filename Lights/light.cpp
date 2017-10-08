#include "light.h"

Light::Light()
    : shadows(true)
#ifdef WCUDA
	, device_ptr(nullptr)
#endif // WCUDA

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

LightCUDA * Light::get_device_ptr()
{
	return device_ptr;
}
