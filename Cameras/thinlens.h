#ifndef THINLENS_H
#define THINLENS_H

#include "camera.h"
#include "pinhole.h"
#include "sampler.h"

class ThinLens : public Pinhole
{
public:
    double _aperture;
    double _focus_distance;
    Sampler* _sampler_ptr;

public:
    ThinLens();
    void set_sampler(Sampler* sampler);

    // Camera interface
public:
    void render_scene(World &w) Q_DECL_OVERRIDE;
	void render_scene_CUDA(World &w);


protected:
    void render_line(ViewPlane vp, int row, World &w);
};

class ThinLensLineRenderer : public QRunnable {
    int _line;
    World* _w;
    ViewPlane _vp;
    ThinLens* _camera;

//    void render_line(ViewPlane vp, int row, World &w);

public:
    ThinLensLineRenderer(const int line, World* w, const ViewPlane& vp, ThinLens* camera);
    // QRunnable interface

    ~ThinLensLineRenderer();
public:
    void run() Q_DECL_OVERRIDE;
};

#endif // THINLENS_H
