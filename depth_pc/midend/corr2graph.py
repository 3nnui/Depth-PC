import numpy as np

from ..frontend.utils import Correspondence
from .graph_gen_depth import GraphGenerator


class Midend(object):
    def __init__(self):
        self.graph_generator_class = GraphGenerator
        self.graph_generator = None

    def get_graph_data(self, corr: Correspondence):
        if corr.tar_img_changed or self.graph_generator is None:
            xy = corr.intrinsic.pixel_to_norm_camera_plane(corr.tar_pos)
            depth = corr.cur_depth
            self.graph_generator = self.graph_generator_class(xy, depth)
        
        missing_kp_indices = np.nonzero(~corr.valid_mask)[0]
        xy = corr.intrinsic.pixel_to_norm_camera_plane(corr.cur_pos_aligned)
        depth = corr.cur_depth
        graph_data = self.graph_generator.get_data(
            current_points=xy,
            depth_cur_norm=depth,
            missing_node_indices=missing_kp_indices
        )
        return graph_data

