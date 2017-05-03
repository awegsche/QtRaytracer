#include "previewworld.h"
#include "PureRandom.h"
#include "ambient.h"
#include "ambientoccluder.h"

PreviewWorld::PreviewWorld(int dowmsampling, int supersampling)
    : m_downsampling(dowmsampling), preview(true), World(), num_samples(supersampling)
{

}

void PreviewWorld::build()
{
    World::build();

    vp.sampler_ptr = new PureRandom(num_samples);
    vp.sampler_ptr->generate_samples();

    auto ao = new AmbientOccluder(.8, .1, 1.0, 1.0, 1.0);
    auto amb = new Ambient(.8, 1,1,1);
    ao->set_sampler(new PureRandom(num_samples));

    render_ambient = ao;
    preview_ambient = ao;


    preview_camera = new Pinhole(*camera_ptr);

    preview_camera->rescale_zoom(1.0f / (float)m_downsampling);
}

void PreviewWorld::render_preview()
{
    preview_camera->render_scene(*this);
}

void PreviewWorld::Keypressed(int key)
{
    switch(key){
    case Qt::Key_A:
        camera_ptr->move_eye_left(1.0);
        break;
    case Qt::Key_W:
        camera_ptr->move_eye_forward(10.0);
        break;
    case Qt::Key_S:
        camera_ptr->move_eye_backward(10.0);
        break;
    case Qt::Key_D:
        camera_ptr->move_eye_right(1.0);
        break;
    case Qt::Key_Up:
        camera_ptr->rotate_down(.05);
        break;
    case Qt::Key_Down:
        camera_ptr->rotate_up(.05);
        break;
    case Qt::Key_Left:
        camera_ptr->rotate_left(.05);
        break;
    case Qt::Key_Right:
        camera_ptr->rotate_right(.05);
        break;

    }

}

void PreviewWorld::run()
{

    if (preview) {
        int old_hres = vp.hres;
        int old_vres = vp.vres;
        ((Pinhole*)camera_ptr)->rescale_zoom(1.0/ (double)m_downsampling);
        vp.hres /= m_downsampling;
        vp.vres /= m_downsampling;
        tracer_ptr->set_shade(true);
        vp.num_samples = 1;
        ambient_ptr = preview_ambient;
        render_camera();
        vp.hres = old_hres;
        vp.vres = old_vres;
        ((Pinhole*)camera_ptr)->rescale_zoom((double)m_downsampling);

    }
    else {
        vp.num_samples = num_samples;
        tracer_ptr->set_shade(false);
        ambient_ptr = render_ambient;
        render_camera();
    }
    preview = true;
    running = true;
}
