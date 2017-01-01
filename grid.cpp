#include "grid.h"
#include "shaderec.h"

Grid::Grid()
    : multiplier(2.0){

}

Grid::Grid(real multi)
    : multiplier(multi){

}

void Grid::setup_cells()
{
    calculate_bounding_box();

    int num_objects = objects.size();
    real wx = boundingbox.x1 - boundingbox.x0;
    real wy = boundingbox.y1 - boundingbox.y0;
    real wz = boundingbox.z1 - boundingbox.z0;
    real s = pow(wx * wy * wz / num_objects, .333333333);

    nx = multiplier * wx / s * 1;
    ny = multiplier * wy / s * 1;
    nz = multiplier * wz / s * 1;

    int num_cells = nx * ny * nz;
    cells.reserve(num_cells);

    for (int j = 0; j < num_cells; j++)
        cells.push_back(nullptr);

    std::vector<int> counts;
    counts.reserve(num_cells);

    for (int j = 0; j < num_cells; j++)
        counts.push_back(0);

    BBox obj_box;
    //int index;

    for(int j = 0; j < num_objects; j++)
    {
        obj_box = objects[j]->get_bounding_box();

        int ixmin = clamp((obj_box.x0 - boundingbox.x0) * nx / (boundingbox.x1 - boundingbox.x0), 0, nx - 1);
        int iymin = clamp((obj_box.y0 - boundingbox.y0) * ny / (boundingbox.y1 - boundingbox.y0), 0, ny - 1);
        int izmin = clamp((obj_box.z0 - boundingbox.z0) * nz / (boundingbox.z1 - boundingbox.z0), 0, nz - 1);
        int ixmax = clamp((obj_box.x1 - boundingbox.x0) * nx / (boundingbox.x1 - boundingbox.x0), 0, nx - 1);
        int iymax = clamp((obj_box.y1 - boundingbox.y0) * ny / (boundingbox.y1 - boundingbox.y0), 0, ny - 1);
        int izmax = clamp((obj_box.z1 - boundingbox.z0) * nz / (boundingbox.z1 - boundingbox.z0), 0, nz - 1);

        for (int iz = izmin; iz <= izmax; iz++)
            for (int iy = iymin; iy <= iymax; iy++)
                for (int ix = ixmin; ix <= ixmax; ix++) {
                     int index = ix + iy * nx + nx * ny * iz;

                    if (counts[index] == 0) {
                        cells[index] = objects[j];
                        counts[index]++;

                    }
                    else if (counts[index] == 1) {
                            Compound* compound_ptr = new Compound;
                            compound_ptr->add_object(cells[index]);
                            compound_ptr->add_object(objects[j]);

                            cells[index] = compound_ptr;
                            counts[index]++;
                        }
                    else {
                        ((Compound*)cells[index])->add_object(objects[j]);
                        counts[index]++;
                    }

                }
    }

    objects.erase(objects.begin(), objects.end());

    for (int j = 0; j < cells.size(); j++)
        if (counts[j] > 1)
            ((Compound*)cells[j])->calculate_bounding_box();
    counts.erase(counts.begin(), counts.end());

}

bool Grid::hit(const Ray &ray, real &t, ShadeRec &sr) const
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


    // traverse the grid

    while (true) {
        GeometricObject* object_ptr = cells[ix + nx * iy + nx * ny * iz];
        t = kHugeValue;
        if (tx_next < ty_next && tx_next < tz_next) {
            if (object_ptr && object_ptr->hit(ray, t, sr) && t < tx_next) {
                //material_ptr = object_ptr->get_material();
                return (true);
            }

            tx_next += dtx;
            ix += ix_step;

            if (ix == ix_stop)
                return (false);
        }
        else {
            if (ty_next < tz_next) {
                if (object_ptr && object_ptr->hit(ray, t, sr) && t < ty_next) {
                    //material_ptr = object_ptr->get_material();
                    return (true);
                }

                ty_next += dty;
                iy += iy_step;

                if (iy == iy_stop)
                    return (false);
            }
            else {
                if (object_ptr && object_ptr->hit(ray, t, sr) && t < tz_next) {
                    //material_ptr = object_ptr->get_material();
                    return (true);
                }

                tz_next += dtz;
                iz += iz_step;

                if (iz == iz_stop)
                    return (false);
            }
        }
    }
}

bool Grid::shadow_hit(const Ray &ray, real &t) const
{
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
    } else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }
    double b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    } else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }
    double c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    } else {
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
    } else {
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
    } else {
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
    } else {
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
    } else {
        tz_next = tz_min + (nz - iz) * dtz;
        iz_step = -1;
        iz_stop = -1;
    }
    if (dz == 0.0) {
        tz_next = kHugeValue;
        iz_step = -1;
        iz_stop = -1;
    }
    // traverse the grid
    while (true) {
        GeometricObject* object_ptr = cells[ix + nx * iy + nx * ny * iz];
        if (tx_next < ty_next && tx_next < tz_next) {
            if (object_ptr && object_ptr->shadow_hit(ray, t) && t < tx_next) {
               // material_ptr = object_ptr->get_material();
                return (true);
            }
            tx_next += dtx;
            ix += ix_step;
            if (ix == ix_stop)
                return (false);
        } else {
            if (ty_next < tz_next) {
                if (object_ptr && object_ptr->shadow_hit(ray, t) && t < ty_next) {
                   // material_ptr = object_ptr->get_material();
                    return (true);
                }
                ty_next += dty;
                iy += iy_step;
                if (iy == iy_stop)
                    return (false);
            } else {
                if (object_ptr && object_ptr->shadow_hit(ray, t) && t < tz_next) {
                   // material_ptr = object_ptr->get_material();
                    return (true);
                }
                tz_next += dtz;
                iz += iz_step;
                if (iz == iz_stop)
                    return (false);
            }
        }
    }

    return false;
}

BBox Grid::get_bounding_box()
{
    return boundingbox;
}
