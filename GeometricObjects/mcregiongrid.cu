#include "mcregiongrid.h"

#ifdef WCUDA




__device__ bool MCRegionGridCUDA::hit(const rayCU & ray, CUDAreal & tmin, ShadeRecCUDA & sr) const
{
	void* mat_ptr = sr.material_ptr;
	CUDAreal ox = ray.o.x;
	CUDAreal oy = ray.o.y;
	CUDAreal oz = ray.o.z;
	CUDAreal dx = ray.d.x;
	CUDAreal dy = ray.d.y;
	CUDAreal dz = ray.d.z;
	CUDAreal x0 = position.x;
	CUDAreal y0 = position.y;
	CUDAreal z0 = position.z;
	CUDAreal x1 = p1.x;
	CUDAreal y1 = p1.y;
	CUDAreal z1 = p1.z;
	CUDAreal tx_min, ty_min, tz_min;
	CUDAreal tx_max, ty_max, tz_max;
	// the following code includes modifications from Shirley and Morley (2003)

	CUDAreal a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	CUDAreal b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	CUDAreal c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	CUDAreal t0, t1;

	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;

	if (tz_min > t0)
		t0 = tz_min;

	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;

	if (tz_max < t1)
		t1 = tz_max;

	if (t0 > t1)
		return(false);


	// initial cell coordinates

	int ix, iy, iz;

	if (inside_bb(position, p1, ray.o)) {  			// does the ray start inside the grid?
		ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
	}
	else {
		CUDAreal3 p = ray.o + t0 * ray.d;  // initial hit point with grid's bounding box
		ix = clamp((p.x - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((p.y - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((p.z - z0) * nz / (z1 - z0), 0, nz - 1);
	}

	// ray parameter increments per cell in the x, y, and z directions

	CUDAreal dtx = (tx_max - tx_min) / nx;
	CUDAreal dty = (ty_max - ty_min) / ny;
	CUDAreal dtz = (tz_max - tz_min) / nz;

	CUDAreal 	tx_next, ty_next, tz_next;
	int 	ix_step, iy_step, iz_step;
	int 	ix_stop, iy_stop, iz_stop;

	if (dx > 0) {
		tx_next = tx_min + (ix + 1) * dtx;
		ix_step = +1;
		ix_stop = nx;
	}
	else {
		tx_next = tx_min + (nx - ix) * dtx;
		ix_step = -1;
		ix_stop = -1;
	}

	if (dx == 0.0) {
		tx_next = kHugeValue;
		ix_step = -1;
		ix_stop = -1;
	}


	if (dy > 0) {
		ty_next = ty_min + (iy + 1) * dty;
		iy_step = +1;
		iy_stop = ny;
	}
	else {
		ty_next = ty_min + (ny - iy) * dty;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dy == 0.0) {
		ty_next = kHugeValue;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dz > 0) {
		tz_next = tz_min + (iz + 1) * dtz;
		iz_step = +1;
		iz_stop = nz;
	}
	else {
		tz_next = tz_min + (nz - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}

	if (dz == 0.0) {
		tz_next = kHugeValue;
		iz_step = -1;
		iz_stop = -1;
	}

	if (tx_next < 0) tx_next = kHugeValue;
	if (ty_next < 0) ty_next = kHugeValue;
	if (tz_next < 0) tz_next = kHugeValue;
	// traverse the grid
	//t = kHugeValue;
	//real t_before = kHugeValue;

	while (true) {
		GeometricObjectCUDA* block_ptr = cells[ix + nx * iy + nx * ny * iz];
		if (tx_next < ty_next && tx_next < tz_next) {
			//real tmin = tx_next - kEpsilon;
			//Material* mptr = sr.material_ptr;
			if (block_ptr && block_ptr->hit(ray, tmin, sr) && tmin < tx_next) {

				return (true);
			}
			//sr.material_ptr = mptr;
			tx_next += dtx;
			ix += ix_step;

			if (ix == ix_stop) {
				sr.material_ptr = mat_ptr;
				return (false);
			}
		}
		else {
			if (ty_next < tz_next) {
				//Material* mptr = sr.material_ptr;
				//real tmin = ty_next - kEpsilon;
				if (block_ptr && block_ptr->hit(ray, tmin, sr) && tmin < ty_next) {
					//material_ptr = object_ptr->get_material();

					return (true);
				}
				//sr.material_ptr = mptr;
				ty_next += dty;
				iy += iy_step;
				//mat_ptr

				if (iy == iy_stop) {
					sr.material_ptr = mat_ptr;
					return (false);
				}
			}
			else {
				//Material* mptr = sr.material_ptr;
				//real tmin = tz_next - kEpsilon;
				//material_ptr = sr.material_ptr;
				if (block_ptr && block_ptr->hit(ray, tmin, sr) && tmin < tz_next) {

					return (true);
				}
				//sr.material_ptr = mptr;
				tz_next += dtz;
				iz += iz_step;

				if (iz == iz_stop) {
					sr.material_ptr = mat_ptr;
					return (false);
				}
			}
		}
	}
}

__device__ bool MCRegionGridCUDA::shadow_hit(const rayCU & ray, CUDAreal & tmin) const
{
	return false;
}
#endif // WCUDA