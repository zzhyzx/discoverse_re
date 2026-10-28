
class BaseConfig:
    mjcf_file_path = ""
    decimation     = 2
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 24,
        "width"  : 1280,
        "height" :  720
    }
    obs_rgb_cam_id = None
    obs_depth_cam_id = None
    rb_link_list   = []
    obj_list       = []
    gs_model_dict  = {}
    use_gaussian_renderer = False
    gs_background_sh_degree = 0
    gs_znear = 0.01  # Metres; Gaussian-centre near clipping and projection plane.
    enable_render = True
    max_render_depth = 5.0
