#include "mcscenerenderer.h"

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
#include "reflective.h"
#include "mcstandardblock.h"
#include "mcwaterblock.h"
#include "mcisoblock.h"
#include "ambientoccluder.h"
#include "pinhole.h"
#include "thinlens.h"
#include "PureRandom.h"
#include "raycast.h"
#include "pointlight.h"
#include <qDebug>

#ifndef WIN64
    const QString texturepath = "/home/awegsche/Minecraft/minecraft/textures/blocks/";
#else
    const QString texturepath = "G:\\Games\\Minecraft\\res\\minecraft\\textures\\blocks\\";
#endif




void MCSceneRenderer::setup_blocklist()
{
    // fill blocklist with nullptr in order to prevent crashes, when an unknown block_id appears
    MCStandardBlock* missing_block = new MCStandardBlock(new Matte(.6,.6, 1, 0, 1));

    for (int i = 0; i < 512; i++)
        blocklist.push_back(missing_block);
    blocklist[0] = nullptr; // air

    blocklist[BlockID::Stone] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "stone.png")));
    blocklist[BlockID::CobbleStone] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "cobblestone.png")));
    blocklist[BlockID::OakWoodPlank] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "planks_oak.png")));
    blocklist[BlockID::Sand] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "sand.png")));
    blocklist[BlockID::Dirt] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "dirt.png")));
    blocklist[BlockID::Stone] = new MCStandardBlock(new Matte(.4, .8, new ImageTexture(texturepath + "stone.png")));

    ImageTexture *oak_leaves = new ImageTexture(texturepath + "leaves_oak.png");
    oak_leaves->colorize(RGBColor(0, 1.0, 0));

    Matte* leaves_mat = new Matte(.4, .8, oak_leaves);
    leaves_mat->has_transparency = true;
    blocklist[BlockID::LeavesOak] = new MCStandardBlock(leaves_mat);

    ImageTexture* grass_top = new ImageTexture(texturepath + "grass_top.png");
    grass_top->colorize(RGBColor(0, 1.0, 0));
    blocklist[BlockID::GrassSide] = new MCIsoBlock(
                new Matte(.5, .8, grass_top),
                new Matte(.5, .8, new ImageTexture(texturepath + "grass_side.png")));
    Matte* logoak_top = new Matte(.5,.8, new ImageTexture(texturepath + "log_oak_top.png"));
    blocklist[BlockID::LogOak] = new MCIsoBlock(
                logoak_top, logoak_top,
                new Matte(.5, .8, new ImageTexture(texturepath + "log_oak.png")));

    Reflective* refl = new Reflective();
    ImageTexture *water_t = new ImageTexture(texturepath + "water_still.png");
    refl->set_diffuse_color(water_t);
    refl->set_ambient_color(water_t);
    refl->set_kr(0.2);
    refl->set_reflective_color(1,1,1);
    refl->set_specular_color(water_t);
    refl->set_ka(.4);
    refl->set_kd(.6);
    refl->set_ks(water_t);
    refl->set_exp(10.0);
    blocklist[BlockID::WaterFlow] = new MCWaterBlock(refl);
    blocklist[BlockID::WaterStill] = new MCWaterBlock(refl);
}

void MCSceneRenderer::switch_to_render()
{
    set_sampler(t_nsamples);
    camera_ptr = m_render_camera;
    ambient_ptr = m_render_ambient;
    tracer_ptr->set_shade(false);

}

void MCSceneRenderer::switch_to_preview()
{
    set_sampler(1);
    camera_ptr = m_render_camera;
    ambient_ptr = m_preview_ambient;
    tracer_ptr->set_shade(true);
}

void MCSceneRenderer::Keypressed(int key)
{
    switch(key){
    case Qt::Key_A:
        m_render_camera->move_eye_left(1.0);
        m_preview_camera->move_eye_left(1.0);
        break;
    case Qt::Key_W:
        m_render_camera->move_eye_forward(10.0);
        m_preview_camera->move_eye_forward(10.0);
        break;
    case Qt::Key_S:
        m_render_camera->move_eye_backward(10.0);
        m_preview_camera->move_eye_backward(10.0);
        break;
    case Qt::Key_D:
        m_render_camera->move_eye_right(1.0);
        m_preview_camera->move_eye_right(1.0);
        break;
    case Qt::Key_Up:
        m_render_camera->rotate_down(.05);
        m_preview_camera->rotate_down(.05);
        break;
    case Qt::Key_Down:
        m_render_camera->rotate_up(.05);
        m_preview_camera->rotate_up(.05);
        break;
    case Qt::Key_Left:
        m_render_camera->rotate_left(.05);
        m_preview_camera->rotate_left(.05);
        break;
    case Qt::Key_Right:
        m_render_camera->rotate_right(.05);
        m_preview_camera->rotate_right(.05);
        break;

    }


}

void MCSceneRenderer::set_sampler(const int n_samples)
{
    vp.sampler_ptr = new PureRandom(n_samples);
    vp.sampler_ptr->generate_samples();
    vp.num_samples = n_samples;

    m_render_ambient->set_sampler(new PureRandom(n_samples));
    m_render_camera->set_sampler(new PureRandom(n_samples));
}

void MCSceneRenderer::set_samples(const int n_samples)
{
	t_nsamples = n_samples;
}

real MCSceneRenderer::get_angle() const
{
    return atan(m_render_camera->get_zoom() / m_render_camera->get_distance() * vp.hres) / GRAD;
}

real MCSceneRenderer::get_vp_distance() const
{
    return m_render_camera->get_distance();
}

void MCSceneRenderer::set_angle(const real angle)
{
    m_render_camera->set_zoom(m_render_camera->get_distance() * tan(angle * GRAD) / vp.hres);
}

void MCSceneRenderer::set_vp_distance(const real distance, const real angle)
{
    m_render_camera->set_distance(distance);
    m_render_camera->set_zoom(distance * tan(angle * GRAD) / vp.hres);

}

void MCSceneRenderer::resize_vp(const int w, const int h)
{
    real frac = (real)w / vp.hres;
    m_render_camera->rescale_zoom(frac);

    vp.hres = w;
    vp.vres = h;
}

void MCSceneRenderer::set_aperture(const real aperture)
{
	m_render_camera->_aperture = aperture;
}


#ifdef WCUDA


void MCSceneRenderer::initCUDADevice()
{
	int device_count;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0)
	{
		qDebug() << "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n";
		
	}

	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		qDebug() << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n";
		return;
	}

	if (deviceProp.major < 1)
	{
		qDebug() << "gpuDeviceInit(): GPU device does not support CUDA.\n";
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(0);
	qDebug() << QString("gpuDeviceInit() CUDA Device [0]: \"%1\n").arg(deviceProp.name);
	

}


#endif // WCUDA

MCSceneRenderer::MCSceneRenderer(QObject *parent)
    : World(parent), t_nsamples(4)
{
    world_grid = new MCRegionGrid();
    world_grid->setup(NREGIONS, 1, NREGIONS, BLOCKLENGTH * 16.0 * 32,
                      Point(-BLOCKLENGTH * 16.0 * 32 * NREGIONS / 2, 0.0, -BLOCKLENGTH * 16.0 * 32 * NREGIONS / 2));


    add_object(world_grid);
    setup_blocklist();

#ifdef WCUDA
	initCUDADevice();
#endif
}

void MCSceneRenderer::add_chunks(MCWorld *world, int x, int y)
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

                        int blockid = (uchar)blocks->_content[j * 256 + k * 16 + i];

                        chunkgrid->addblock(i, j, k, blockid);
                    }
            chunkgrid->set_parent(grid, this);
            grid->addblock(chunk->x, Y, chunk->y, chunkgrid);
        }

    }
    world_grid->addblock(x + NREGIONS / 2, 0, y + NREGIONS / 2, grid);

}

void MCSceneRenderer::run()
{
    running = true;
    render_camera();
}

void MCSceneRenderer::build()
{
    m_preview_ambient = new Ambient(1.5, 1, 1, 1);
    m_render_ambient = new AmbientOccluder(2.0, 0.1, 1.0, 1.0, 1.0);
    m_preview_camera = new Pinhole(250, 100, 250, 500, 0, 250, 100, 1.0);
    m_render_camera = new ThinLens(250, 100, 250, 500, 0, 250, 100, 10.0, 0.1);

    vp.hres = 640;
    vp.vres = 480;

    add_light(new PointLight(3.0, 1.0, 1.0, 0.2, 10000, 50000, 2000));




    tracer_ptr = new RayCast(this);

    set_sampler(4);

    //preview_camera->rescale_zoom(1.0f / (float)m_downsampling);

    this->background_color = RGBColor(.8, .9, 1.0);

    switch_to_preview();

    set_vp_distance(100.0, 45.0);

}
