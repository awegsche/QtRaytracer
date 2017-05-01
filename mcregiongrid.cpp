#include "mcregiongrid.h"
#include "matte.h"
#include "shaderec.h"

MCRegionGrid::MCRegionGrid()
{

}



void MCRegionGrid::setup(int nx_, int ny_, int nz_, real unit, Point pos)
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
        cells.push_back(nullptr);
}

void MCRegionGrid::addblock(int x, int y, int z, GeometricObject *block)
{
    cells[x + nx * y + nx * ny * z] = block;
}

bool MCRegionGrid::hit(const Ray &ray, real &t, ShadeRec &sr) const
{
    Material* mat_ptr = sr.material_ptr;
    double ox = ray.o.X;
    double oy = ray.o.Y;
    double oz = ray.o.Z;
    double dx = ray.d.X;
    double dy = ray.d.Y;
    double dz = ray.d.Z;
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
        ix = clamp((p.X - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((p.Y - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((p.Z - z0) * nz / (z1 - z0), 0, nz - 1);
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

    if (tx_next < 0) tx_next = kHugeValue;
    if (ty_next < 0) ty_next = kHugeValue;
    if (tz_next < 0) tz_next = kHugeValue;
    // traverse the grid
    t = kHugeValue;
    real t_before = kHugeValue;

    while (true) {
        GeometricObject* block_ptr = cells[ix + nx * iy + nx * ny * iz];
        if (tx_next < ty_next && tx_next < tz_next) {
            real tmin = tx_next - kEpsilon;
            //Material* mptr = sr.material_ptr;
            if (block_ptr && block_ptr->hit(ray, t_before, sr) && tmin < t) {
                t = tmin;

                t = tx_next;
                return (true);
            }
            //sr.material_ptr = mptr;
            t_before = tx_next;
            tx_next += dtx;
            ix += ix_step;
            sr.normal = Normal(-(real)ix_step, 0, 0);
            sr.hdir = ix_step > 0 ? ShadeRec::South : ShadeRec::North;
            sr.t_Before = t_before;

            if (ix == ix_stop) {
                sr.material_ptr = mat_ptr;
                return (false);
            }
        }
        else {
            if (ty_next < tz_next) {
                //Material* mptr = sr.material_ptr;
                real tmin = ty_next - kEpsilon;
                if (block_ptr && block_ptr->hit(ray, t_before, sr) && tmin < t) {
                    //material_ptr = object_ptr->get_material();
                    t=tmin;
                    t = ty_next;
                    return (true);
                }
                //sr.material_ptr = mptr;
                t_before = ty_next;
                ty_next += dty;
                iy += iy_step;
                sr.normal = Normal(0.0, -(real)iy_step, 0);
                sr.hdir = iy_step > 0 ? ShadeRec::Bottom : ShadeRec::Top;
                sr.t_Before = t_before;
                //mat_ptr

                if (iy == iy_stop) {
                    sr.material_ptr = mat_ptr;
                    return (false);
                }
            }
            else {
                //Material* mptr = sr.material_ptr;
                real tmin = tz_next - kEpsilon;
                //material_ptr = sr.material_ptr;
                if (block_ptr && block_ptr->hit(ray, tmin, sr) && tmin < t) {
                    //material_ptr = object_ptr->get_material();
                    //sr.material_ptr = material_ptr;
                    t=tmin;
                   // t = tz_next;
                    return (true);
                }
                //sr.material_ptr = mptr;
                t_before = tz_next;
                tz_next += dtz;
                iz += iz_step;
                sr.normal = Normal(0.0, 0.0, -(real)iz_step);
                sr.hdir = iz_step > 0 ? ShadeRec::West : ShadeRec::East;
                sr.t_Before = t_before;

                if (iz == iz_stop) {
                    sr.material_ptr = mat_ptr;
                    return (false);
                }
            }
        }
    }
}

bool MCRegionGrid::shadow_hit(const Ray &ray, real &t) const
{
    Material* mat_ptr;
    double ox = ray.o.X;
    double oy = ray.o.Y;
    double oz = ray.o.Z;
    double dx = ray.d.X;
    double dy = ray.d.Y;
    double dz = ray.d.Z;
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
        ix = clamp((p.X - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((p.Y - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((p.Z - z0) * nz / (z1 - z0), 0, nz - 1);
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

    if (tx_next < 0) tx_next = kHugeValue;
    if (ty_next < 0) ty_next = kHugeValue;
    if (tz_next < 0) tz_next = kHugeValue;
    // traverse the grid
    t = kHugeValue;
    real t_before = kHugeValue;

    while (true) {
        GeometricObject* block_ptr = cells[ix + nx * iy + nx * ny * iz];
        if (tx_next < ty_next && tx_next < tz_next) {
            real tmin = tx_next - kEpsilon;
            if (block_ptr && block_ptr->shadow_hit(ray, tmin) && tmin < t) {

                //material_ptr = object_ptr->get_material();
                t = tx_next;


                return (true);
            }
            t_before = tx_next;
            tx_next += dtx;
            ix += ix_step;

            if (ix == ix_stop)
                return (false);
        }
        else {
            if (ty_next < tz_next) {
                real tmin = ty_next - kEpsilon;
                if (block_ptr && block_ptr->shadow_hit(ray, tmin) && tmin < t) {
                    //material_ptr = object_ptr->get_material();
                    t = ty_next;

                    return (true);
                }

                t_before = ty_next;
                ty_next += dty;
                iy += iy_step;

                if (iy == iy_stop)
                    return (false);
            }
            else {
                real tmin = tz_next - kEpsilon;
                if (block_ptr && block_ptr->shadow_hit(ray, tmin) && tmin < t) {
                    //material_ptr = object_ptr->get_material();

                    t = tz_next;
                    return (true);
                }
                t_before = tz_next;
                tz_next += dtz;
                iz += iz_step;

                if (iz == iz_stop)
                    return (false);
            }
        }
    }
}

BBox MCRegionGrid::get_bounding_box()
{
    return boundingbox;
}
