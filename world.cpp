#include "world.h"
#include "shaderec.h"
#include "point2d.h"
#include "camera.h"
#include "ambient.h"
#include "nbttag.h"
#include "mcgrid.h"
#include "nbttagcompound.h"
#include "nbttaglist.h"
#include "nbttagint.h"
#include "nbttagbyte.h"
#include "nbttagbytearray.h"
#include "matte.h"
#include "constants.h"
#include "textureholder.h"
#include "matte.h"
#include "mcregiongrid.h"
//#include "simple_scene.cpp"


World::World(QObject *parent)
    : QThread(parent), camera_ptr(nullptr), ambient_ptr(new Ambient), running(true)
{
    world_grid = new Grid();
    this->tholder = new TextureHolder();
    setup_blocklist(tholder);
}


void World::setup_blocklist(TextureHolder *th)
{
    Matte* mat = new Matte(0.4, 0.5, 1.0, .0, 1.0);
    blocklist.insert(0, nullptr);
    for (int i = 1; i < 256; i++)
    {
        MCBlock* block = new MCBlock();
        Texture* sidetext = th->get_side(i);
        if (sidetext != nullptr) {
            Matte* matside = new Matte(.4, .8, 0,0,0);
            matside->set_color(sidetext);
            block->mat_side = matside;
        }
        else
            block->mat_side = mat;

        Texture* toptext = th->get_top(i);
        if (toptext != nullptr) {
            Matte* mattop = new Matte(.4,.8,0,0,0);
            mattop->set_color(toptext);
            block->mat_top = mattop;
        }
        else
            block->mat_top = mat;
        blocklist.insert(i, block);
    }
}

void World::add_object(GeometricObject *o)
{
    objects.push_back(o);
}

void World::add_light(Light *l)
{
    lights.push_back(l);
}

void World::add_chunks(MCWorld* world, int x, int y)
{
    real X = BLOCKLENGTH * x * 32 * 16;
    real Z = BLOCKLENGTH * y * 32 * 16;
    Point p0(X, 0.0, Z);
    MCRegionGrid *grid = new MCRegionGrid();
    grid->setup(32, 16, 32, BLOCKLENGTH * 16.0, p0);

    for(Chunk* chunk : world->_chunks) {
        //Chunk* chunk = world->_chunks[i];
        NBTTag* nbtchunk = chunk->root;
        if (nbtchunk->ID() == NBTTag::TAG_End) return;
        NBTTagList<NBTTagCompound> *regions = static_cast<NBTTagList<NBTTagCompound> *>(nbtchunk->get_child("Level")->get_child("Sections"));



        for (NBTTagCompound* region : regions->_children)
        {
            int Y = ((NBTTagByte*)region->get_child("Y"))->getValue();

            MCGrid* chunkgrid = new MCGrid();
            chunkgrid->setup(16, 16, 16, BLOCKLENGTH, Point(X + chunk->x * 16, Y * 16, Z + chunk->y * 16));

            NBTTagByteArray* blocks = ((NBTTagByteArray*)region->get_child("Blocks"));
            for(int j = 0; j < 16; j++)
                for (int k = 0; k < 16; k++)
                    for (int i = 0; i < 16; i++)
                    {

                        int blockid = blocks->_content[j * 256 + k * 16 + i];

                        chunkgrid->addblock(i, j, k, blockid);
                    }
            chunkgrid->set_parent(grid);
            grid->addblock(chunk->x, Y, chunk->y, chunkgrid);
        }

    }
    grid->blocklist = &this->blocklist;
    add_object(grid);

}

void World::render_scene_()
{
    RGBColor pixel_color;
    Ray ray;
    real zw = 1000.0;
    double x,y;
    int n = (int)sqrt((float)vp.num_samples);
    //Point2D pp;

    ray.d = Vector(0,0,-1.0);

    for(int row = 0; row < vp.vres; row++) {
        for(int column = 0; column < vp.hres; column++) {
            pixel_color = RGBColor(0,0,0);

            for (int p = 0; p < n; p++)
                for (int q = 0; q < n; q++) {


                    x = vp.s * (column - 0.5 * vp.hres + ((float) rand()) / (float) RAND_MAX);
                    y = vp.s * (row - 0.5 * vp.vres + ((float) rand()) / (float) RAND_MAX);
                    ray.o = Point(x,y,zw);
                    pixel_color += tracer_ptr->trace_ray(ray, 0);
                }
            pixel_color /= (float)vp.num_samples;
            emit display_pixel(row,column, (int)(pixel_color.r * 255.0), (int)(pixel_color.g * 255.0),(int)(pixel_color.b * 255.0) );
        }
    }
    emit done();
}

void World::render_camera()
{
    camera_ptr->render_scene(*this);
}

//void World::render_scene() const
//{

//}

ShadeRec World::hit_bare_bones_objects(const Ray &ray)
{
    ShadeRec sr(this);
    real t;
    real tmin = kHugeValue;
    int num_objects = objects.size();

    for (int j = 0; j < num_objects; j++)
        if (objects[j]->hit(ray, t, sr) && (t < tmin)) {
            sr.hit_an_object = true;
            tmin = t;
            //sr.color = objects[j]->get_color();
        }
    return sr;
}

ShadeRec World::hit_objects(const Ray &ray)
{
    ShadeRec sr(this);
    real t = kHugeValue;
    Normal normal;
    Point local_hit_point;
    real tmin = kHugeValue;
    Material* mat_ptr;
    int num_objects = objects.size();

    for(int j = 0; j < num_objects; j++)
        if (objects[j]->hit(ray, t, sr) && t < tmin) {
            sr.hit_an_object = true;
            tmin = t;
            //sr.material_ptr = objects[j]->get_material(); // moved to hit function
            sr.hitPoint = ray.o + t * ray.d;
            mat_ptr = sr.material_ptr;
            normal = sr.normal;
            local_hit_point = sr.local_hit_point;
        }
    if (sr.hit_an_object) {
        sr.t = tmin;
        sr.normal = normal;
        sr.local_hit_point = local_hit_point;
        sr.material_ptr = mat_ptr;
    }

    return sr;
}

void World::dosplay_p(int r, int c, const RGBColor& pixel_color)
{
    RGBColor color = pixel_color.truncate();

    emit display_pixel(r, c, (int)(color.r * 255.0), (int)(color.g * 255.0),(int)(color.b * 255.0));
}

//Pixel World::display_p(Pixel& result, const Pixel &p)
//{
//    RGBColor color = p.color.truncate();
//    result = p;
//    emit p.w->display_pixel(p.point.Y, p.point.X, (int)(color.r * 255.0), (int)(color.g * 255.0),(int)(color.b * 255.0));
//    return p;
//}

void World::run()
{
    render_camera();
}
