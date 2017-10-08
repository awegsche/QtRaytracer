#include "mcblock.h"
#include "shaderec.h"
#include "constants.h"
#include <math.h>

MCBlock::MCBlock()
{

}

MCBlockCUDA * MCBlock::get_device_ptr() 
{	
	if (device_ptr)
		return (MCBlockCUDA*)device_ptr;

	cudaMallocManaged(&device_ptr, sizeof(MCBlockCUDA));;
	return (MCBlockCUDA*)device_ptr;
}
