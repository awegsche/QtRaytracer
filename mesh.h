#ifndef MESH_H
#define MESH_H

#include <vector>

#include "point.h"
#include "normal.h"

class Mesh
{
public:
    std::vector<Point> vertices;
    std::vector<Normal> normals;
    std::vector<real> u;
    std::vector<real> v;
    std::vector<std::vector<int>> vertex_faces;
    int num_vertices;
    int num_triangles;

public:
    Mesh();

    Mesh(const Mesh& m);

    Mesh& operator= (const Mesh& rhs);
};

#endif // MESH_H
