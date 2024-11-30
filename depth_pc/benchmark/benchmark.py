import os
import pickle
import time
from typing import Dict, Optional, Union

import numpy as np
import torch

from depth_pc.utils.depth_anything_v2.dpt import DepthAnythingV2

from .environment import BenchmarkEnvAffine, BenchmarkEnvRender
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .stop_policy import ErrorHoldingStopPolicy

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

class BaseRecord(object):
    def __init__(self, folder):
        self.folder = folder
        self.trajs = []
        self.rates = []
        self.timings = dict()
        self.errors = []

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
    
    def clear(self):
        self.trajs.clear()
        self.rates.clear()
        self.errors.clear()
        self.timings.clear()

    def append_traj(
        self, 
        cur_bcT: np.ndarray, 
        vel: np.ndarray, 
        timing: Optional[Dict] = None, 
        error: float = "Unknown",
        **kwargs
    ):
        self.trajs.append(cur_bcT)
        self.rates.append(vel)
        self.errors.append(error)

        if timing is None:
            timing = {"timestamp": time.time()}

        for k in timing:
            if k not in self.timings:
                self.timings[k] = []
            self.timings[k].append(timing[k])
    
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        trajs = np.stack(self.trajs, axis=0)
        rates = np.stack(self.rates, axis=0)
        errors = np.array(self.errors)
        timings = {k: np.array(v) for k, v in self.timings.items()}
        fname = os.path.join(self.folder, fname)

        save_dict = dict(
            tar_bcT=tar_bcT,  # (4, 4)
            ini_bcT=ini_bcT,  # (4, 4)
            trajs=trajs,  # (N, 4, 4)
            rates=rates,  # (N, 6),
            errors=errors,  # (N,)
            **timings
        )

        return fname, save_dict


class NpzRecord(BaseRecord):
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        fname, save_dict = super().finalize(fname, tar_bcT, ini_bcT)
        np.savez(fname, **save_dict)
        print("[INFO] Result saved to {}".format(fname))
        self.clear()


class PickleRecord(BaseRecord):
    def __init__(self, folder):
        super().__init__(folder)
        self.extra_data_traj = []
    
    def clear(self):
        self.extra_data_traj.clear()
        return super().clear()
    
    def append_traj(
        self, 
        cur_bcT: np.ndarray, 
        vel: np.ndarray, 
        timing: Optional[Dict] = None, 
        error: float = "Unknown",
        extra_data = None
    ):  
        if len(self.extra_data_traj) and extra_data is not None:
            if isinstance(extra_data, dict):
                extra_data["tar_img"] = None
            else:
                setattr(extra_data, "tar_img", None)
            # avoid saving duplicate target image
        self.extra_data_traj.append(extra_data)
        return super().append_traj(cur_bcT, vel, timing, error)
    
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        fname, save_dict = super().finalize(fname, tar_bcT, ini_bcT)
        save_dict["extra_data_traj"] = self.extra_data_traj
        with open(fname, "wb") as fp:
            pickle.dump(save_dict, fp)
        print("[INFO] Result saved to {}".format(fname))
        self.clear()


def run_benchmark(
    env: Union[BenchmarkEnvRender, BenchmarkEnvAffine],
    pipeline: Union[CorrespondenceBasedPipeline, ImageBasedPipeline],
    stop_policy: ErrorHoldingStopPolicy,
    result_folder: str,
    record: bool = False,
    skip_saved: bool = True,
):
    npz_recorder = NpzRecord(result_folder) if record else None
    if skip_saved and result_folder and record:
        env.skip_indices(env.get_global_indices_of_saved(result_folder))
    
    requires_intrinsic = isinstance(pipeline, CorrespondenceBasedPipeline)
    need_rgb_channel_shuffle = pipeline.REQUIRE_IMAGE_FORMAT != env.RETURN_IMAGE_FORMAT
    print("Need to reorder RGB channel: {}".format(need_rgb_channel_shuffle))
    
    for i in range(len(env)):
        env.clear_debug_items()
        if isinstance(env, BenchmarkEnvRender):
            tar_img, tar_depth = env.init(i)
        else:
            tar_img = env.init(i)
        if need_rgb_channel_shuffle:
            tar_img = np.ascontiguousarray(tar_img[:, :, ::-1])
        tPo_norm = np.linalg.norm(env.target_wcT[:3, 3])

        if requires_intrinsic:
            pipeline.frontend.reset_intrinsic(env.camera.intrinsic)
        pipeline.set_target(tar_img,tar_depth, dist_scale=tPo_norm, intrinsic=env.camera.intrinsic)
        stop_policy.reset()

        while True:
            if isinstance(env, BenchmarkEnvRender):
                cur_img, cur_depth = env.observation()
            else:
                cur_img = env.observation()
            if need_rgb_channel_shuffle:
                cur_img = np.ascontiguousarray(cur_img[:, :, ::-1])
            
            vel, data, timing = pipeline.get_control_rate(cur_img, cur_depth)
            need_stop = (
                stop_policy(data, env.steps*env.dt) or 
                env.exceeds_maximum_steps() or
                data is None
                or env.camera_under_ground()
                # or env.abnormal_pose()
            )

            if record:
                npz_recorder.append_traj(
                    env.current_wcT, vel, timing, stop_policy.cur_err)

            if need_stop:
                break
            
            env.action(vel * 2)

        print("[INFO] Round: {}/{}".format(i+1, len(env)))
        print("[INFO] Steps: {}/{}".format(env.steps, env.max_steps))
        env.print_pose_err()
        print("---------------------------------------------------")

        if record:
            npz_recorder.finalize(
                env.prefer_result_fname(i),
                env.target_wcT, 
                env.initial_wcT
            )
