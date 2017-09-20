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


};

#endif // THINLENS_H
