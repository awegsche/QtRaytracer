#include "world.h"
#include "shaderec.h"
#include "cuda_runtime.h"

__global__
void kernel_hit_objects(int n, float* x, float* y) {

}

extern "C" ShadeRec * hit_test()
{
	float *x, *y;
	int N = 1000;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the GPU
	kernel_hit_objects<<<1, 1>>> (N, x, y);

	cudaDeviceSynchronize();


	cudaFree(x);
	cudaFree(y);

	return nullptr;
}