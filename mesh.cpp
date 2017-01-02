#include "mesh.h"

Mesh::Mesh()
{

}

Mesh::Mesh(const Mesh &m)
    :   vertices(m.vertices),
        normals(m.normals),
        u(m.u),
        v(m.v),
        num_triangles(m.num_triangles),
        num_vertices(m.num_vertices)
{

}

Mesh &Mesh::operator=(const Mesh &rhs)
{
    if (this == &rhs)
        return (*this);

    vertices 		= rhs.vertices;
    normals  		= rhs.normals;
    u  				= rhs.u;
    v  				= rhs.v;
    num_vertices	= rhs.num_vertices;
    num_triangles	= rhs.num_triangles;

    return (*this);

}
