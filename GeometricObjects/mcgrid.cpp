#include "mcgrid.h"
#include "matte.h"
#include "shaderec.h"
#include "mcscenerenderer.h"

MCGrid::MCGrid()
    : parent(nullptr)
{

}



void MCGrid::setup(int nx_, int ny_, int nz_, real unit, Point pos)
{
    m_unit = unit;
    position = pos;
    nx = nx_;
    ny = ny_;
    nz = nz_;

    p1 = pos + Vector((real)nx * unit, (real)ny * unit, (real)nz * unit);


    boundingbox = BBox(position, p1);
    int count_ = nx * ny * nz;
    cells.reserve(nx * ny * nz);
    for (int i = 0; i < count_; i++)
        cells.push_back(0);
}

void MCGrid::read_nbt(QString filename, World *w)
{
    // normally: load nbt file here.
    // for testing: fill the grid randomly

    setup(10, 10, 10, BLOCKLENGTH, Point(0,0,0));

    cells[1 + nx * 0 + nx * ny * 1] = 2;
    cells[3 + nx * 0 + nx * ny * 1] = 2;
    cells[2 + nx * 0 + nx * ny * 3] = 2;
    cells[2 + nx * 0 + nx * ny * 4] = 1;
    cells[0 + nx * 0 + nx * ny * 4] = 17;
}

void MCGrid::addblock(int x, int y, int z, int block)
{
    cells[x + nx * y + nx * ny * z] = block;
}

void MCGrid::set_parent(MCRegionGrid *grid, MCSceneRenderer *w)
{
    parent = grid;
    _w = w;
}

bool MCGrid::hit(const Ray &ray, real &t, ShadeRec &sr) const
{
    Material* mat_ptr = sr.material_ptr;
    double ox = ray.o.X();
    double oy = ray.o.Y();
    double oz = ray.o.Z();
    double dx = ray.d.X();
    double dy = ray.d.Y();
    double dz = ray.d.Z();
    double x0 = boundingbox.x0;
    double y0 = boundingbox.y0;
    double z0 = boundingbox.z0;
    double x1 = boundingbox.x1;
    double y1 = boundingbox.y1;
    double z1 = boundingbox.z1;
    double tx_min, ty_min, tz_min;
    double tx_max, ty_max, tz_max;
    // the following code includes modifications from Shirley and Morley (2003)

    double a = 1.0 / dx;
    if (a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    double b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    double c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    double t0, t1;

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

    if (boundingbox.inside(ray.o)) {  			// does the ray start inside the grid?
        ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
    }
    else {
        Point p = ray.o + t0 * ray.d;  // initial hit point with grid's bounding box
        ix = clamp((p.X() - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((p.Y() - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((p.Z() - z0) * nz / (z1 - z0), 0, nz - 1);
    }

    // ray parameter increments per cell in the x, y, and z directions

    double dtx = (tx_max - tx_min) / nx;
    double dty = (ty_max - ty_min) / ny;
    double dtz = (tz_max - tz_min) / nz;

    double 	tx_next, ty_next, tz_next;
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

//    if (tx_next < 0) tx_next = kHugeValue;
//    if (ty_next < 0) ty_next = kHugeValue;
//    if (tz_next < 0) tz_next = kHugeValue;



        // Test if there is a block face glued to the bounding box:

        MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];
        Point block_p0 = Point(x0 + nx * BLOCKLENGTH, y0 + ny * BLOCKLENGTH, z0 + nz * BLOCKLENGTH);
        if (block_ptr) {
            real t_before = kHugeValue;

            real tx_min_pp = tx_next - dtx;
            real ty_min_pp = ty_next - dty;
            real tz_min_pp = tz_next - dtz;

            if (ix != 0 && ix != (nx - 1)) tx_min_pp = -kHugeValue;
            if (iy != 0 && iy != (ny - 1)) ty_min_pp = -kHugeValue;
            if (iz != 0 && iz != (nz - 1)) tz_min_pp = -kHugeValue;


            if (tx_min_pp > ty_min_pp && tx_min_pp > tz_min_pp) {
                sr.normal = Normal(-(real)ix_step, 0, 0);
                sr.hdir = ix_step > 0 ? ShadeRec::South : ShadeRec::North;
                t_before = tx_min_pp;
            }
            else if (ty_min_pp > tz_min_pp) {
                sr.normal = Normal(0, -(real)iy_step, 0);
                sr.hdir = iy_step > 0 ? ShadeRec::Bottom : ShadeRec::Top;
                t_before = ty_min_pp;

            }
            else   {
                sr.normal = Normal(0, 0, -(real)iz_step);
                sr.hdir = iz_step > 0 ? ShadeRec::West : ShadeRec::East;
                t_before = tz_min_pp;

            }
            if (block_ptr->block_hit(ray, block_p0, t_before, sr)) {
                t = t_before;


                return (true);
            }
        }



    // traverse the grid
    t = kHugeValue;
    real t_before = kHugeValue;

    while (true) {
//        MCBlock* block_ptr = cells[ix + nx * iy + nx * ny * iz];

        if (tx_next < ty_next && tx_next < tz_next) {
            //real tmin = tx_next - kEpsilon;
            //Material* mptr = sr.material_ptr;
            sr.normal = Normal(-(real)ix_step, 0, 0);
            sr.hdir = ix_step > 0 ? ShadeRec::South : ShadeRec::North;
            sr.t_Before = t_before;
            t_before = tx_next;
            tx_next += dtx;
            ix += ix_step;
            if (ix == ix_stop) {
                sr.material_ptr = mat_ptr;
                return (false);
            }


            MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];
            Point block_p0 = Point(x0 + nx * BLOCKLENGTH, y0 + ny * BLOCKLENGTH, z0 + nz * BLOCKLENGTH);

            if (block_ptr && block_ptr->block_hit(ray, block_p0, t_before, sr))  {
                t = t_before;


                return (true);
            }
            //sr.material_ptr = mptr;

        }
        else {
            if (ty_next < tz_next) {
                //Material* mptr = sr.material_ptr;
                sr.normal = Normal(0.0, -(real)iy_step, 0);
                sr.hdir = iy_step > 0 ? ShadeRec::Bottom : ShadeRec::Top;
                sr.t_Before = t_before;
                t_before = ty_next;
                ty_next += dty;
                iy += iy_step;
                if (iy == iy_stop) {
                    sr.material_ptr = mat_ptr;
                    return (false);
                }

                MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];
                Point block_p0 = Point(x0 + nx * BLOCKLENGTH, y0 + ny * BLOCKLENGTH, z0 + nz * BLOCKLENGTH);


                if (block_ptr && block_ptr->block_hit(ray, block_p0, t_before, sr)) {
                    //material_ptr = object_ptr->get_material();
                    t=t_before;
                    //t = ty_next;
                    return (true);
                }
                //sr.material_ptr = mptr;
                //mat_ptr

            }
            else {
                //Material* mptr = sr.material_ptr;
                sr.normal = Normal(0.0, 0.0, -(real)iz_step);
                sr.hdir = iz_step > 0 ? ShadeRec::West : ShadeRec::East;
                sr.t_Before = t_before;
                t_before = tz_next;
                tz_next += dtz;
                iz += iz_step;
                if (iz == iz_stop) {
                    sr.material_ptr = mat_ptr;
                    return (false);
                }

                MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];
                Point block_p0 = Point(x0 + nx * BLOCKLENGTH, y0 + ny * BLOCKLENGTH, z0 + nz * BLOCKLENGTH);


                //material_ptr = sr.material_ptr;
                if (block_ptr && block_ptr->block_hit(ray, block_p0, t_before, sr))  {
                    //material_ptr = object_ptr->get_material();
                    //sr.material_ptr = material_ptr;
                    t=t_before;
                   // t = tz_next;
                    return (true);
                }
                //sr.material_ptr = mptr;

            }
        }
    }
}

bool MCGrid::shadow_hit(const Ray &ray, real &t) const
{

    double ox = ray.o.X();
    double oy = ray.o.Y();
    double oz = ray.o.Z();
    double dx = ray.d.X();
    double dy = ray.d.Y();
    double dz = ray.d.Z();
    double x0 = boundingbox.x0;
    double y0 = boundingbox.y0;
    double z0 = boundingbox.z0;
    double x1 = boundingbox.x1;
    double y1 = boundingbox.y1;
    double z1 = boundingbox.z1;
    double tx_min, ty_min, tz_min;
    double tx_max, ty_max, tz_max;
    // the following code includes modifications from Shirley and Morley (2003)

    double a = 1.0 / dx;
    if (a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    double b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    double c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    double t0, t1;

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

    if (boundingbox.inside(ray.o)) {  			// does the ray start inside the grid?
        ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
    }
    else {
        Point p = ray.o + t0 * ray.d;  // initial hit point with grid's bounding box
        ix = clamp((p.X() - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((p.Y() - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((p.Z() - z0) * nz / (z1 - z0), 0, nz - 1);
    }

    // ray parameter increments per cell in the x, y, and z directions

    double dtx = (tx_max - tx_min) / nx;
    double dty = (ty_max - ty_min) / ny;
    double dtz = (tz_max - tz_min) / nz;

    double 	tx_next, ty_next, tz_next;
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

//    if (tx_next < 0) tx_next = kHugeValue;
//    if (ty_next < 0) ty_next = kHugeValue;
//    if (tz_next < 0) tz_next = kHugeValue;



        // Test if there is a block face glued to the bounding box:

        MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];

        if (block_ptr) {
            real t_before = 0;

            real tx_min_pp = tx_next - dtx;
            real ty_min_pp = ty_next - dty;
            real tz_min_pp = tz_next - dtz;

            if (ix != 0 && ix != (nx - 1)) tx_min_pp = -kHugeValue;
            if (iy != 0 && iy != (ny - 1)) ty_min_pp = -kHugeValue;
            if (iz != 0 && iz != (nz - 1)) tz_min_pp = -kHugeValue;


            if (tx_min_pp > ty_min_pp && tx_min_pp > tz_min_pp) {

                t_before = tx_min_pp;
            }
            else if (ty_min_pp > tz_min_pp) {
                t_before = ty_min_pp;

            }
            else   {
                t_before = tz_min_pp;

            }
            if (block_ptr->shadow_hit(ray, t_before)) {

                return (true);
            }
        }



    // traverse the grid
    t = kHugeValue;
    real t_before = kHugeValue;

    while (true) {
//        MCBlock* block_ptr = cells[ix + nx * iy + nx * ny * iz];

        if (tx_next < ty_next && tx_next < tz_next) {
            t_before = tx_next;
            tx_next += dtx;
            ix += ix_step;
            if (ix == ix_stop) {
                return (false);
            }


            MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];

            if (block_ptr && block_ptr->shadow_hit(ray, t_before)) {
                t=t_before;
                return (true);
            }
            //sr.material_ptr = mptr;

        }
        else {
            if (ty_next < tz_next) {
                //Material* mptr = sr.material_ptr;
                t_before = ty_next;
                ty_next += dty;
                iy += iy_step;
                if (iy == iy_stop) {
                    return (false);
                }

                MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];

                if (block_ptr && block_ptr->shadow_hit(ray, t_before)) {
                    //material_ptr = object_ptr->get_material();
                    t=t_before;
                    //t = ty_next;
                    return (true);
                }
                //sr.material_ptr = mptr;
                //mat_ptr

            }
            else {
                //Material* mptr = sr.material_ptr;
                t_before = tz_next;
                tz_next += dtz;
                iz += iz_step;
                if (iz == iz_stop) {
                    return (false);
                }


                MCBlock* block_ptr = _w->blocklist[cells[ix + nx * iy + nx * ny * iz]];

                //material_ptr = sr.material_ptr;
                if (block_ptr && block_ptr->shadow_hit(ray, t_before)) {
                    //material_ptr = object_ptr->get_material();
                    //sr.material_ptr = material_ptr;
                    t=t_before;
                   // t = tz_next;
                    return (true);
                }
                //sr.material_ptr = mptr;

            }
        }
    }
}

BBox MCGrid::get_bounding_box()
{
    return boundingbox;
}

#ifdef WCUDA

MCGridCUDA * MCGrid::get_device_ptr() 
{
	
	if (device_ptr)
		return (MCGridCUDA*)device_ptr;

	size_t numcells = cells.size();
	size_t memsize =sizeof(MCGridCUDA);
	MCGridCUDA* gr = (MCGridCUDA*)device_ptr;

	cudaMallocManaged(&gr, memsize);
	cudaMallocManaged(&gr->cells, sizeof(int) * numcells);

	gr->nx = nx;
	gr->ny = ny;
	gr->nz = nz;
	gr->num_cells = numcells;

	for (int j = 0; j < numcells; j++)
		gr->cells[j] = cells[j];

	gr->p0 = __make_CUDAreal3(position.X(), position.Y(), position.Z());
	gr->p1 = __make_CUDAreal3(p1.X(), p1.Y(), p1.Z());
	
	return gr;
}

#endif