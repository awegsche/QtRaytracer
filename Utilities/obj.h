#ifndef OBJ_H
#define OBJ_H

// Loads wavefront OBJ using Qt classes for the IO

#include "mesh.h"
#include <QFile>


void get_vertices (Mesh* mesh_ptr, QFile& infile);

#endif // OBJ_H
