#ifndef THINLENS_H
#define THINLENS_H

#include "camera.h"
#include "pinhole.h"
#include "sampler.h"
#include "ray.h" 


class ThinLens : public Pinhole
{
public:
    real _aperture;
   
    Sampler* _sampler_ptr;

public:
    ThinLens();
    ThinLens(const real eye_x, const real eye_y, const real eye_z,
             const real lookat_x, const real lookat_y, const real lookat_z,
             const real distance, const real zoom_, const real aperture);
    void set_sampler(Sampler* sampler);

    // Camera interface
public:
    void render_scene(World &w) Q_DECL_OVERRIDE;

	Ray get_click_ray(const real vpx, const real vpy, const ViewPlane& vp) Q_DECL_OVERRIDE;

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
