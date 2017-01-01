#ifndef GRID_H
#define GRID_H

#include "compound.h"
#include "vector"

class Grid : public Compound
{
private:
    std::vector<GeometricObject*> cells;
    int nx, ny, nz;
    real multiplier;
//    Point min_coordinates();
//    Point max_coordinates();

public:
    Grid();
    Grid(real multi);

    void setup_cells();
    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &t, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &t) const;
    BBox get_bounding_box();
};

#endif // GRID_H
