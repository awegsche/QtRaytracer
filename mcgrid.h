#ifndef MCGRID_H
#define MCGRID_H

#include "compound.h"
#include "vector"
#include "mesh.h"
#include "mcblock.h"
#include <QString>
#include "world.h"
#include "mcregiongrid.h"

// MCGrid only contains Blocks. To create a hierarchical Grid, use
// MCRegionsGrid

class MCGrid : public Compound
{
private:
    std::vector<int> cells;
    int nx, ny, nz;
    real multiplier;
    Point position;
    Point p1;
    MCRegionGrid* parent;
    real m_unit;

public:
    MCGrid();

    void setup(int nx_, int ny_, int nz_, real unit, Point pos);

    void read_nbt(QString filename, World *w);
    void addblock(int x, int y, int z, int block);
    void set_parent(MCRegionGrid *grid);

    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &t, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &tmin) const;
    BBox get_bounding_box();
};

#endif // MCGRID_H
