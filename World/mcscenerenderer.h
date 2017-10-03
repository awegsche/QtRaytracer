#ifndef MCSCENERENDERER_H
#define MCSCENERENDERER_H

#ifdef WCUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif // WCUDA

#include "world.h"

class Pinhole;
class ThinLens;
class Ambient;
class AmbientOccluder;


// A grid of regoins with edge length NREGIONS will be created
const int NREGIONS = 32;



class MCSceneRenderer : public World
{
    Q_OBJECT
private:

    MCRegionGrid* world_grid;

    Pinhole* m_preview_camera;
    ThinLens* m_render_camera;

    Ambient* m_preview_ambient;
    AmbientOccluder* m_render_ambient;

    int t_nsamples;

#ifdef WCUDA

	void initCUDADevice();

#endif // WCUDA

	

public:
    std::vector<MCBlock*> blocklist;
    bool preview;


public:
    MCSceneRenderer(QObject* parent = nullptr);

    void setup_blocklist();

    void switch_to_render();
    void switch_to_preview();
    void Keypressed(int key);
    void set_sampler(const int n_samples);
	void set_samples(const int n_samples);

    real get_angle() const;
    real get_vp_distance() const;
    void set_angle(const real angle);
    void set_vp_distance(const real distance, const real angle);
    void resize_vp(const int w, const int h);
	void set_aperture(const real aperture);

    ShadeRec hit_objects(const Ray& ray);
    ShadeRec* hit_objects_CUDA();

signals:
	void stdLog(const QString &message);

public:

    void add_chunks(MCWorld* world, int x, int y);


//signals:
//    void display_pixel(int row, int column, int r, int g, int b);
//    void display_line(const int row, const uint* rgb);
//    void done();

    // QThread interface
protected:
    void run();

    // World interface
public:
    void build() Q_DECL_OVERRIDE;
};



#endif // MCSCENERENDERER_H
