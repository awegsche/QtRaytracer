#ifndef GRID_H
#define GRID_H

#include "compound.h"
#include "vector"
#include "mesh.h"

class Grid : public Compound
{
private:
    std::vector<GeometricObject*> cells;
    int nx, ny, nz;
    real multiplier;
    Mesh* mesh_ptr;
    bool reverse_normal;

//    Point min_coordinates();
//    Point max_coordinates();

public:
    Grid();
    Grid(real multi);

    void setup_cells();

    void
    read_ply_file(char* file_name, const int triangle_type);

    void compute_mesh_normals();


    // GeometricObject interface
public:
    bool hit(const Ray &ray, real &t, ShadeRec &sr) const;
    bool shadow_hit(const Ray &ray, real &t) const;
    BBox get_bounding_box();
};

#endif // GRID_H
